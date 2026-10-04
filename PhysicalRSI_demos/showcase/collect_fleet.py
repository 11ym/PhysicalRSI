"""Distribute a saved layout batch across the configured DSW hosts."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cycle',type=Path,required=True)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--python',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--ssh-key',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.cycle/'cycle.json').read_text())
    root=args.cycle/f'round-{args.round:04d}'
    layouts=json.loads((root/'layouts.json').read_text())['layouts']
    hosts=plan['hosts'];count=len(layouts)
    code=root/'code';code.mkdir(exist_ok=True)
    collector=code/'collect_dexjoco.py'
    if collector.exists():raise ValueError('This collection round already has a collector snapshot')
    shutil.copyfile(Path(__file__).with_name('collect_dexjoco.py'),collector)
    assignments=[];offset=0
    for rank,host in enumerate(hosts):
        size=count//len(hosts)+(rank<count%len(hosts))
        seeds=[row['seed'] for row in layouts[offset:offset+size]];offset+=size
        if seeds!=list(range(seeds[0],seeds[0]+len(seeds))):raise ValueError('Expected contiguous layout seeds')
        assignments.append({'host':host,'rank':rank,'seeds':seeds,'state':'pending'})
    (root/'assignments.json').write_text(json.dumps(assignments,indent=2))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',str(args.ssh_key)]
    def worker(row):
        target=root/f"worker-{row['rank']:02d}";target.mkdir(exist_ok=False)
        status=target/'worker.json'
        row={**row,'state':'checking'};status.write_text(json.dumps(row))
        try:
            check=subprocess.run(ssh+['root@'+row['host'],'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],capture_output=True,text=True,timeout=30,check=True)
            if check.stdout.strip():raise RuntimeError('GPU is busy')
            command=['env','MUJOCO_GL=egl','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=1',
                     'OPENBLAS_NUM_THREADS=1','PYTHONUNBUFFERED=1',args.python,str(collector),
                     '--root',str(args.source),'--output',str(target/'data'),
                     '--seed',str(row['seeds'][0]),'--episodes',str(len(row['seeds'])),
                     '--chunk-size','30','--sample-every','10','--layout-manifest',str(root/'layouts.json')]
            row['state']='running';status.write_text(json.dumps(row))
            with (target/'execution.log').open('w') as log:
                result=subprocess.run(ssh+['root@'+row['host'],shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,timeout=10800)
            row.update(state='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
        except Exception as error:row.update(state='failed',error=str(error))
        status.write_text(json.dumps(row,indent=2))
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        pending={pool.submit(worker,row) for row in assignments}
        completed=[]
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
            completed.extend(future.result() for future in done)
            episodes=[]
            for path in root.glob('worker-*/data/episode-*/result.json'):
                try:episodes.append(json.loads(path.read_text()))
                except (OSError,ValueError):continue
            successful={row['seed'] for row in episodes if row['success']}
            summary={'layouts':count,'recorded':len(episodes),'successful':len(successful),
                     'active_workers':len(pending),'finished_workers':completed,
                     'missing_successful_layouts':[row['seed'] for row in layouts if row['seed'] not in successful]}
            (root/'collection.json').write_text(json.dumps(summary,indent=2))
            print(f"Layouts {len(episodes)}/{count} · successful {len(successful)} · active workers {len(pending)}",flush=True)
    if len(successful)!=count:raise SystemExit('Some layouts still need successful demonstrations')


if __name__=='__main__':main()
