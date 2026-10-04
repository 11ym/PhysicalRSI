"""Fine-tune SmolVLA on successful visual action chunks collected in Dexjoco."""
import argparse
from io import BytesIO
import json
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='lerobot/smolvla_base')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('Training steps and batch size must be positive')
    import numpy as np
    import torch
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    manifest = json.loads((args.dataset/'dataset.json').read_text())
    arrays = []
    for episode in manifest['episodes']:
        if episode['success']:
            with np.load(args.dataset/f"episode-{episode['seed']}"/'transitions.npz') as data:
                arrays.append({k:data[k].copy() for k in ('images','states','actions')})
    if not arrays:
        raise ValueError('The dataset contains no successful demonstrations')
    images, states, actions = (np.concatenate([row[k] for row in arrays])
                              for k in ('images','states','actions'))
    state_mean, state_std = states.mean(0), states.std(0).clip(min=1e-3)
    action_mean, action_std = actions.mean((0,1)), actions.std((0,1)).clip(min=1e-3)
    config = PreTrainedConfig.from_pretrained(args.model)
    config.input_features = {
        'observation.images.front':PolicyFeature(FeatureType.VISUAL,(3,256,256)),
        'observation.state':PolicyFeature(FeatureType.STATE,(states.shape[-1],)),
    }
    config.output_features = {'action':PolicyFeature(FeatureType.ACTION,(actions.shape[-1],))}
    config.chunk_size = actions.shape[1]
    config.n_action_steps = config.chunk_size
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(args.model, config=config).to(config.device).train()
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer([manifest['instruction']]*args.batch_size, padding='max_length',
                       truncation=True, max_length=config.tokenizer_max_length, return_tensors='pt')
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    normalization = BytesIO()
    np.savez(normalization, state_mean=state_mean, state_std=state_std,
             action_mean=action_mean, action_std=action_std)
    (args.output/'normalization.npz').write_bytes(normalization.getvalue())
    print(f'SmolVLA training · {len(images)} samples · {config.device}', flush=True)
    rng = np.random.default_rng(0)
    with (args.output/'metrics.jsonl').open('w') as log:
        for step in range(args.steps):
            indices = rng.integers(len(images), size=args.batch_size)
            def tensor(value):return torch.as_tensor(value,device=config.device)
            batch = {
                'observation.images.front':tensor(images[indices]).permute(0,3,1,2).float()/255,
                'observation.state':tensor((states[indices]-state_mean)/state_std),
                'action':tensor((actions[indices]-action_mean)/action_std),
                OBS_LANGUAGE_TOKENS:tokens['input_ids'].to(config.device),
                OBS_LANGUAGE_ATTENTION_MASK:tokens['attention_mask'].bool().to(config.device),
            }
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):raise RuntimeError('Training loss is not finite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0)
            optimizer.step()
            row={'step':step+1,'loss':float(loss.detach())}
            log.write(json.dumps(row)+'\n');log.flush()
            print(f"Step {step+1}/{args.steps} · loss {row['loss']:.5f}",flush=True)
    with tempfile.TemporaryDirectory(prefix='physicalrsi-vla-export-') as directory:
        checkpoint = Path(directory)/'policy'
        policy.save_pretrained(checkpoint)
        config_path = checkpoint/'config.json'
        saved_config = json.loads(config_path.read_text())
        saved_config['type'] = config.type
        config_path.write_text(json.dumps(saved_config, indent=2))
        shutil.copytree(checkpoint, args.output/'policy', copy_function=shutil.copyfile)
    (args.output/'training.json').write_text(json.dumps({
        'model':args.model,'steps':args.steps,'samples':len(images),
        'instruction':manifest['instruction'],'action_chunk_size':config.chunk_size,
        'normalization':'normalization.npz','scope':'Fine-tuning run; task success requires rollout evaluation',
    },indent=2))
    print('Saved policy and normalization to '+str(args.output),flush=True)


if __name__ == '__main__':
    main()
