"""Evaluate a pi05 checkpoint on saved Dexjoco layouts and record RGB rollouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--layouts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--base-model', action='store_true')
    parser.add_argument('--max-steps', type=int, default=1200)
    args = parser.parse_args()
    source = args.root/'openpi'
    sys.path[:0] = [str(source/'src'), str(source/'packages/openpi-client/src'), str(args.root/'dexjoco')]
    os.chdir(source)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import jax.numpy as jnp
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from openpi.models import model, pi0_config
    from openpi.policies.policy import Policy
    from openpi.policies.single_arm_policy import SingleArmInputs, SingleArmOutputs
    from openpi.training.config import ModelTransformFactory
    from openpi.shared import normalize
    from openpi import transforms
    from dexjoco.sim.envs.panda_click_mouse_env import PandaClickMouseGymEnv

    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=30, max_token_len=250,
        paligemma_variant='gemma_2b' if args.base_model else 'gemma_2b_lora',
        action_expert_variant='gemma_300m' if args.base_model else 'gemma_300m_lora')
    policy_model = model_config.load(model.restore_params(args.checkpoint/'params', dtype=jnp.bfloat16))
    stats = normalize.load(args.data)
    policy = Policy(policy_model, transforms=[SingleArmInputs(model_config.model_type),
        transforms.Normalize(stats, use_quantiles=True), *ModelTransformFactory()(model_config).inputs],
        output_transforms=[transforms.Unnormalize(stats, use_quantiles=True), SingleArmOutputs()])
    layouts = json.loads(args.layouts.read_text())['layouts'][args.rank::args.workers]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for layout in layouts:
        seed = layout['seed']
        destination = args.output/f'episode-{seed}'
        destination.mkdir()
        env = PandaClickMouseGymEnv(render_mode='rgb_array', randomize=False,
                                   randomize_dynamics=False, seed=seed, hz=100000)
        writer = None
        success = False
        steps = 0
        outcome = {}
        noise = np.random.default_rng(seed)
        try:
            env._compute_observation = lambda: {}
            env.reset()
            np.testing.assert_allclose(env.data.qpos, layout['qpos'], rtol=0, atol=1e-9)
            actions = []
            for step in range(args.max_steps):
                observation = type(env)._compute_observation(env) if not actions or step % 10 == 0 else None
                if step % 10 == 0:
                    frame = np.ascontiguousarray(observation['images']['front'], dtype=np.uint8)
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '5', '-i', '-', '-an',
                            '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(destination/'rollout.mp4')], stdin=subprocess.PIPE)
                    writer.stdin.write(frame.tobytes())
                if not actions:
                    tcp = np.asarray(observation['state']['tcp_pose']).ravel()
                    state = np.concatenate((tcp[:3], Rotation.from_quat(tcp[[4,5,6,3]]).as_rotvec(),
                                            np.asarray(observation['state']['gripper_pose']).ravel())).astype(np.float32)
                    inputs = {'state': state, 'prompt': 'Place the mouse on the mousepad and press its left button.',
                        'base': np.asarray(Image.fromarray(observation['images']['ego_right']).resize((256,256))),
                        'wrist': np.asarray(Image.fromarray(observation['images']['wrist']).resize((256,256)))}
                    chunk = policy.infer(inputs, noise=noise.standard_normal((30,32)).astype(np.float32))['actions']
                    if not np.isfinite(chunk).all():
                        raise ValueError('Policy produced non-finite actions')
                    actions = list(chunk)
                action = actions.pop(0)
                quat = Rotation.from_rotvec(action[3:6]).as_quat()
                raw_action = np.concatenate((action[:3], quat[[3,0,1,2]], action[6:22]))
                _, _, terminated, truncated, info = env.step(raw_action)
                steps = step+1
                success = bool(info.get('succeed', False))
                if success or terminated or truncated:
                    break
            outcome = {'mouse_on_pad': bool(env._mouse_in_mousepad()),
                       'click_registered': bool(env._display_blue)}
        finally:
            env.close()
            if writer is not None:
                writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError('Video encoding failed')
        result = {'seed': seed, 'success': success, 'steps': steps,
                  'checkpoint': str(args.checkpoint), 'outcome': outcome}
        (destination/'result.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        (args.output/'evaluation.json').write_text(json.dumps({'episodes': results,
            'success_rate': sum(row['success'] for row in results)/len(results)}, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
