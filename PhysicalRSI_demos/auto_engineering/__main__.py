import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description='Self-built Isaac Sim lab: sample transfer and evidence-driven repair')
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--isaac-python',default=os.environ.get('PHYSICALRSI_SIM_PYTHON'))
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args(argv)
    if args.report_only:
        from .report import render
        print(render(args.workspace));return 0
    if not args.isaac_python:p.error('Pass --isaac-python or set PHYSICALRSI_SIM_PYTHON (Isaac Sim 5.1)')
    from .harness import run
    print(json.dumps(run(args.workspace,args.isaac_python,args.episodes,args.gui),indent=2))
    print('Open '+str(args.workspace.resolve()/'index.html'))
    return 0

if __name__=='__main__':raise SystemExit(main())
