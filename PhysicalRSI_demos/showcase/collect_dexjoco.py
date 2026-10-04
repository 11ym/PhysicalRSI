"""Collect visual action chunks from the installed Dexjoco mouse teacher."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--layout-manifest', type=Path)
    parser.add_argument('--teacher', default='recover_200')
    parser.add_argument('--teacher-parameters', type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.chunk_size < 1 or args.sample_every < 1:
        parser.error('Episode count and chunk size must be positive')
    sys.path[:0] = [str(args.root.resolve()), str((args.root/'dexjoco').resolve())]
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    from PIL import Image
    from rsi.click_mouse_datagen import Candidate, CANDIDATES, ROUND2_CANDIDATES, ROUND3_CANDIDATES, ROUND4_CANDIDATES, run_episode

    args.output.mkdir(parents=True, exist_ok=True)
    teachers = {candidate.name: candidate for candidate in
                (*CANDIDATES, *ROUND2_CANDIDATES, *ROUND3_CANDIDATES, *ROUND4_CANDIDATES)}
    if args.teacher not in teachers:
        parser.error('Unknown teacher: ' + args.teacher)
    teacher = teachers[args.teacher]
    if args.teacher_parameters:
        teacher = Candidate(**json.loads(args.teacher_parameters.read_text()))
    instruction = 'Place the mouse on the mousepad and press its left button.'
    reports = []
    layouts = {row['seed']:row for row in json.loads(args.layout_manifest.read_text())['layouts']} if args.layout_manifest else {}
    for seed in range(args.seed, args.seed+args.episodes):
        images, states, indices, actions = [], [], [], []
        ego_images, wrist_images = [], []
        destination = args.output/f'episode-{seed}'
        destination.mkdir()
        writer = None
        temporary = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        video_path = Path(temporary.name)
        temporary.close()
        def observe(env, action):
            nonlocal writer
            if action is None and args.layout_manifest:
                expected = layouts[seed]
                if not np.allclose(env.data.qpos, expected['qpos'], rtol=0, atol=1e-9):
                    raise ValueError('Reset does not reproduce the saved layout')
            if action is not None:
                actions.append(np.asarray(action, dtype=np.float32).copy())
            if len(actions) % args.sample_every:
                return
            obs = type(env)._compute_observation(env)
            frame = np.ascontiguousarray(obs['images']['front'], dtype=np.uint8)
            if writer is None:
                height, width = frame.shape[:2]
                writer = subprocess.Popen([
                    'ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{width}x{height}','-r',str(1/env.control_dt/args.sample_every),'-i','-',
                    '-an','-c:v','libx264','-preset','fast','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(video_path)], stdin=subprocess.PIPE)
            writer.stdin.write(frame.tobytes())
            images.append(np.asarray(Image.fromarray(frame).resize((256,256))))
            ego_images.append(np.asarray(Image.fromarray(obs['images']['ego_right']).resize((256,256))))
            wrist_images.append(np.asarray(Image.fromarray(obs['images']['wrist']).resize((256,256))))
            states.append(np.concatenate([obs['state']['tcp_pose'].ravel(),
                                          obs['state']['gripper_pose'].ravel()]).astype(np.float32))
            indices.append(len(actions))
        print(f'Collecting click_mouse · seed {seed}', flush=True)
        try:
            report = run_episode(teacher, seed, int(os.environ.get('PHYSICALRSI_ENV_GPU','0')), observe)
        finally:
            if writer:
                writer.stdin.close()
                if writer.wait() != 0:
                    video_path.unlink(missing_ok=True)
                    raise RuntimeError('Rollout video encoding failed')
                shutil.copyfile(video_path, destination/'rollout.mp4')
            video_path.unlink(missing_ok=True)
        usable = [i for i, start in enumerate(indices) if start < len(actions)]
        if report['success']:
            chunks = np.array([[actions[min(indices[i]+j,len(actions)-1)]
                                for j in range(args.chunk_size)] for i in usable])
            buffer = BytesIO()
            np.savez_compressed(buffer,
                                images=np.array(images)[usable], states=np.array(states)[usable],
                                actions=chunks, instruction=instruction,
                                ego_images=np.array(ego_images)[usable], wrist_images=np.array(wrist_images)[usable])
            (destination/'transitions.npz').write_bytes(buffer.getvalue())
        (destination/'result.json').write_text(json.dumps(report, indent=2))
        reports.append({'seed':seed, 'success':report['success'], 'steps':report['steps'],
                        'samples':len(usable) if report['success'] else 0})
        print(f"Episode {seed}: success={report['success']} · samples={reports[-1]['samples']}", flush=True)
    manifest = {'task':'click_mouse','instruction':instruction,'chunk_size':args.chunk_size,
                'source':'simulator-assisted demonstration generation',
                'sample_every':args.sample_every,'control_dt':0.02,
                'student_inputs':['RGB','robot proprioception','instruction'], 'episodes':reports}
    (args.output/'dataset.json').write_text(json.dumps(manifest, indent=2))
    print('Dataset written to '+str(args.output), flush=True)
    if not any(r['success'] for r in reports):
        raise SystemExit('No successful demonstrations collected')


if __name__ == '__main__':
    main()
