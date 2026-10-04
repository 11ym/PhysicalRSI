"""Use a configured local CLI model to produce bounded workbench intents."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
from types import SimpleNamespace
from PhysicalRSI_core.infra.language import LanguageModel


class LocalModel:
    config = SimpleNamespace(protocol='chat_completions')
    tool_result = LanguageModel.tool_result

    def structured(self, prompt, schema):
        with tempfile.TemporaryDirectory(prefix='physicalrsi-intent-') as directory:
            root=Path(directory);output=root/'answer.json';definition=root/'schema.json'
            definition.write_text(json.dumps(schema))
            command=[os.environ['PHYSICALRSI_SHOWCASE_AGENT_CLI'],'exec','--ephemeral',
                     '--skip-git-repo-check','--sandbox','read-only','--ignore-user-config',
                     '--output-schema',str(definition),'--output-last-message',str(output),'-C',directory]
            if os.environ.get('PHYSICALRSI_SHOWCASE_AGENT_MODEL'):
                command.extend(['-m',os.environ['PHYSICALRSI_SHOWCASE_AGENT_MODEL']])
            result=subprocess.run(command,input=prompt,text=True,capture_output=True,timeout=180)
            if result.returncode or not output.is_file():
                raise RuntimeError('Local model did not return a completed intent')
            return json.loads(output.read_text())

    def complete(self, history, tools):
        schema = {'type':'object','properties':{
            'answer':{'type':'string'},
            'action':{'type':'string','enum':['none','layouts','collect','train','cycle']},
            'count':{'type':'integer','minimum':0}},
            'required':['answer','action','count'],'additionalProperties':False}
        prompt=('Interpret the conversation as a workbench assistant. Do not use tools, read files, '
                'or execute commands. Return only the structured answer. action=none means reply only; '
                'layouts/collect/train/cycle requests will be executed by the application. '
                'cycle starts or follows the distributed pi05 loop; count is the number of rounds, '
                'with zero meaning continuous operation. Never claim a job has run.\n'
                +json.dumps(history,ensure_ascii=False))
        intent=self.structured(prompt,schema)
        action=intent['action'];answer=intent['answer']
        if action=='none':return answer,[],[{'role':'assistant','content':answer}]
        if action not in ('layouts','collect','train','cycle') or type(intent['count']) is not int:
            raise ValueError('Local model returned an invalid action')
        call={'id':uuid.uuid4().hex,'name':'workbench_run',
              'arguments':json.dumps({'action':action,'count':intent['count']})}
        native={'role':'assistant','content':answer,'tool_calls':[
            {'id':call['id'],'type':'function','function':{'name':call['name'],'arguments':call['arguments']}}]}
        return answer,[call],[native]
