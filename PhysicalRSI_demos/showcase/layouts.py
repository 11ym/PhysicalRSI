"""Save newly sampled Dexjoco layouts before demonstration collection."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--seed',type=int,default=1000000)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new layout manifest for each round')
    sys.path.insert(0,str(args.root/'dexjoco'))
    os.environ.setdefault('MUJOCO_GL','egl')
    from dexjoco.sim.envs.panda_click_mouse_env import PandaClickMouseGymEnv
    layouts=[]
    for seed in range(args.seed,args.seed+args.count):
        env=PandaClickMouseGymEnv(render_mode='rgb_array',randomize=False,
                                  randomize_dynamics=False,seed=seed,hz=100000)
        try:
            env._compute_observation=lambda:{}
            env.reset()
            layouts.append({'seed':seed,'qpos':env.data.qpos.tolist(),
                            'qvel':env.data.qvel.tolist(),'mouse_pose':env.mouse_ori_pose.tolist(),
                            'mocap_pos':env.data.mocap_pos.tolist(),
                            'mocap_quat':env.data.mocap_quat.tolist()})
        finally:env.close()
        if len(layouts)%10==0:print(f'Sampled {len(layouts)}/{args.count} layouts',flush=True)
    if len({tuple(row['mouse_pose']) for row in layouts})!=len(layouts):
        raise ValueError('Layout sampling produced duplicate mouse placements')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'task':'click_mouse','environment':{
        'randomize':False,'randomize_dynamics':False,'control_dt':0.02},'layouts':layouts},indent=2))
    print('Saved layout manifest: '+str(args.output),flush=True)


if __name__=='__main__':main()
