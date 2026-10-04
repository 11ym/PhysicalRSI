"""Run a recorded RoboDojo episode with the installed PhysicalRSI adapter."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import time



def stop_server(server):
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')',1)[1].split()
            processes[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    owned = {server.pid}
    while True:
        children = {pid for pid,(parent,_) in processes.items() if parent in owned}
        if children <= owned:break
        owned.update(children)
    def signal_owned(value):
        for pid in owned:
            try:
                fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
                if pid in processes and fields[19] == processes[pid][1]:
                    os.kill(pid,value)
            except (OSError,IndexError):pass
    signal_owned(signal.SIGTERM)
    try:server.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    time.sleep(.2)
    signal_owned(signal.SIGKILL)
    server.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    framework = args.framework.resolve()
    benchmark = framework.parent
    policy = framework/'policy/physicalRSI'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PHYSICALRSI_EVAL_OUTPUT=str(args.output/'policy'),
               ROBODOJO_RESULT_ROOT=str(args.output/'rollouts'), EVAL_NUM='1',
               PYTHONPATH=os.pathsep.join([str(policy/'runtime'),str(benchmark),str(framework),env.get('PYTHONPATH','')]),
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.3')
    server_command=['bash', str(policy/'setup_eval_policy_server.sh'), 'RoboDojo',args.task,
                    'skill-library','arx_x5','joint','0',env.get('PHYSICALRSI_POLICY_GPU','0'),
                    env['PHYSICALRSI_POLICY_PYTHON'],str(port),'127.0.0.1']
    server=subprocess.Popen(server_command,cwd=framework,env=env,start_new_session=True)
    try:
        deadline=time.monotonic()+1200
        while True:
            if server.poll() is not None:raise RuntimeError('Policy server exited during startup')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Policy server startup timed out')
                time.sleep(1)
        print('Policy ready · starting one recorded episode',flush=True)
        command=[env['PHYSICALRSI_SIM_PYTHON'],'-u','src/eval_client/main.py',
                 '--task_name',args.task,'--env_cfg_type','arx_x5','--device_id',env.get('PHYSICALRSI_ENV_GPU','0'),
                 '--policy_name','physicalRSI','--port',str(port),'--host','127.0.0.1','--protocol','ws',
                 '--additional_info','ckpt_name=skill-library,action_type=joint','--seed','0',
                 '--headless','--enable_cameras','--num_envs','1',
                 '--kit_args','--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera']
        result=subprocess.run(command,cwd=benchmark,env=env)
        raise SystemExit(result.returncode)
    finally:
        stop_server(server)


if __name__=='__main__':main()
