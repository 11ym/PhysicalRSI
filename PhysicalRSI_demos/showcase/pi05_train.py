"""Train pi05 LoRA on Dexjoco demonstrations with one JAX process per DSW."""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


class Demonstrations:
    def __init__(self, directory, transform, norm_stats, replay_directories=()):
        self.directory = directory
        self.manifest = json.loads((directory/'dataset.json').read_text())
        self.episodes = []
        normalization = json.loads((directory/'norm_stats.json').read_text())
        for source in (directory, *replay_directories):
            if json.loads((source/'norm_stats.json').read_text()) != normalization:
                raise ValueError('Replay datasets must use the same normalization')
            for episode in json.loads((source/'dataset.json').read_text())['episodes']:
                self.episodes.append({**episode, 'directory': str((source/episode['directory']).resolve())})
        self.ends = np.cumsum([row['samples'] for row in self.episodes])
        self.transform = transform
        self.norm_stats = norm_stats
        self.arrays = {}

    def sample(self, index):
        episode = int(np.searchsorted(self.ends, index, side='right'))
        offset = int(index - (self.ends[episode-1] if episode else 0))
        row = self.episodes[episode]
        if episode not in self.arrays:
            self.arrays[episode] = {name: np.load(self.directory/row['directory']/(name+'.npy'), mmap_mode='r')
                                    for name in ('base', 'wrist', 'state', 'actions')}
        data = {name: np.array(value[offset]) for name, value in self.arrays[episode].items()}
        data['prompt'] = row['instruction']
        return self.transform(data)

    def data_config(self):
        from openpi.training.config import DataConfig
        return DataConfig(asset_id='dexjoco', norm_stats=self.norm_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--replay-data', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--process-count', type=int, default=1)
    parser.add_argument('--process-id', type=int, default=0)
    parser.add_argument('--coordinator')
    args = parser.parse_args()
    source = args.root/'openpi'
    sys.path[:0] = [str(source/'src'), str(source/'packages/openpi-client/src'), str(source/'scripts')]
    os.chdir(source)
    import jax
    if args.process_count > 1:
        jax.distributed.initialize(coordinator_address=args.coordinator, num_processes=args.process_count,
                                   process_id=args.process_id, local_device_ids=[0], initialization_timeout=300)
    from jax.experimental import multihost_utils
    from openpi import transforms
    from openpi.models import model, pi0_config
    from openpi.policies.single_arm_policy import SingleArmInputs
    from openpi.shared import normalize
    from openpi.training import config, optimizer, sharding, checkpoints, weight_loaders
    from train import init_train_state, train_step

    if args.batch_size % args.process_count:
        raise ValueError('Global batch size must divide evenly across DSWs')
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=30, max_token_len=250,
                                      paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    training = config.TrainConfig(name='dexjoco-pi05', exp_name=args.output.name, model=model_config,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.checkpoint)),
        freeze_filter=model_config.get_freeze_filter(), ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(100, max(1, args.steps//10)),
                                                  peak_lr=5e-5, decay_steps=max(3, args.steps)),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.), batch_size=args.batch_size,
        num_train_steps=args.steps, wandb_enabled=False)
    stats = normalize.load(args.data)
    transform = transforms.compose([SingleArmInputs(model_config.model_type),
        transforms.Normalize(stats, use_quantiles=True), *config.ModelTransformFactory()(model_config).inputs])
    dataset = Demonstrations(args.data, transform, stats, args.replay_data)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if args.process_id == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'training.json').write_text(json.dumps({
            'model': 'pi05', 'adaptation': 'LoRA', 'initial_checkpoint': str(args.checkpoint),
            'dataset': str(args.data), 'replay_datasets': [str(path) for path in args.replay_data],
            'samples': int(dataset.ends[-1]), 'steps': args.steps, 'global_batch_size': args.batch_size,
            'processes': args.process_count, 'state': 'initializing'}, indent=2))
    multihost_utils.sync_global_devices('training-output-ready')
    manager, _ = checkpoints.initialize_checkpoint_dir(args.output/'checkpoints', keep_period=None,
                                                        overwrite=False, resume=True)
    print(f'Initializing pi05 on rank {args.process_id}/{args.process_count}', flush=True)
    rng, init_rng = jax.random.split(jax.random.key(42))
    state, state_sharding = init_train_state(training, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    if args.process_id == 0:
        record = json.loads((args.output/'training.json').read_text())
        record['state'] = 'training'
        (args.output/'training.json').write_text(json.dumps(record, indent=2))
    step_fn = jax.jit(functools.partial(train_step, training),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    local_batch = args.batch_size//args.process_count
    generator = np.random.default_rng(42)
    order = np.empty(0, dtype=np.int64)
    start = time.monotonic()
    for step in range(args.steps):
        while len(order) < args.batch_size:
            order = np.concatenate((order, generator.permutation(dataset.ends[-1])))
        indices, order = order[:args.batch_size], order[args.batch_size:]
        indices = indices[args.process_id*local_batch:(args.process_id+1)*local_batch]
        samples = [dataset.sample(index) for index in indices]
        batch = jax.tree.map(lambda *values: np.stack(values), *samples)
        batch = jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), batch)
        observations = model.Observation.from_dict(batch)
        with sharding.set_mesh(mesh):
            state, info = step_fn(rng, state, (observations, batch['actions']))
        if step % 10 == 0 or step == args.steps-1:
            metrics = {name: float(value.addressable_data(0)) for name, value in info.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError('Non-finite training metrics')
            if args.process_id == 0:
                row = {'step': step+1, 'elapsed_seconds': time.monotonic()-start, **metrics}
                with (args.output/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
        if (step+1) % 500 == 0 or step == args.steps-1:
            checkpoints.save_state(manager, state, dataset, step+1)
    manager.wait_until_finished()
    if args.process_id == 0:
        record = json.loads((args.output/'training.json').read_text())
        record.update(state='completed', checkpoint=str(args.output/'checkpoints'/str(args.steps)))
        (args.output/'training.json').write_text(json.dumps(record, indent=2))
    multihost_utils.sync_global_devices('training-completed')
    if args.process_count > 1:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()
