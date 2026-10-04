"""Conversation tools for the local demonstration workbench."""
import json
import os
from pathlib import Path
from PhysicalRSI_core.infra.language import LanguageModel, ModelConfig
from .cycle_status import cycle_status


class DemoConversation:
    def __init__(self, jobs):
        self.jobs = jobs
        self.history = []

    def ask(self, message):
        path = os.environ.get('PHYSICALRSI_SHOWCASE_MODEL_CONFIG')
        if os.environ.get('PHYSICALRSI_SHOWCASE_AGENT_CLI'):
            from .local_model import LocalModel
            model = LocalModel()
        elif path:
            model = LanguageModel(ModelConfig(**json.loads(Path(path).read_text())))
        else:
            raise ValueError('Configure a CLI model or PHYSICALRSI_SHOWCASE_MODEL_CONFIG; slash commands also work without a model')
        if not self.history:
            self.history.append({'role':'system', 'content':
                'You help a user collect Dexjoco demonstrations and train a VLA. '
                'The installed collection task is click_mouse: place a mouse on its pad and press the left button. '
                'The simulator-assisted teacher creates RGB, robot-state and action-chunk data; '
                'only successful episodes are used for SmolVLA fine-tuning. '
                'Use the workbench_run tool only for actions the user requests. '
                'Choose one job per response. Generate and finish layouts before collecting the matching episode count. '
                'Default to two layouts/episodes or 100 training steps when no count is given. '
                'Training uses the latest successful collection in this session. '
                'For pi05 or repeated 100-layout collection/training/evaluation, use action cycle. '
                'For cycle, count means rounds (default two); zero means continue until paused. '
                'It follows an already-running training round rather than restarting it. '
                'Job startup does not prove completion or task success. '
                'Explain unavailable tasks briefly and never invent execution results. '
                'Respond in the user language. Never request credentials in chat.'})
        self.history.append({'role':'user','content':message})
        tool = {'name':'workbench_run','description':'Start collection, small VLA fine-tuning, or the distributed pi05 cycle.',
                'parameters':{'type':'object','properties':{
                    'action':{'type':'string','enum':['layouts','collect','train','cycle']},
                    'count':{'type':'integer','minimum':0}},
                    'required':['action','count'],'additionalProperties':False}}
        with self.jobs.lock:
            states = [{k:job.get(k) for k in ('id','kind','status','returncode')}
                      for job in list(self.jobs.jobs.values())[-4:]]
        context = {'role':'system','content':'Current process states (completed means process exit, not task proficiency): '+json.dumps(states)+
                   '\nSeparately recorded pi05 cycle progress: '+json.dumps(cycle_status())+
                   '\nThe collect/train actions run the small SmolVLA demonstration; cycle runs the distributed pi05 loop.'}
        answer, calls, native = model.complete([*self.history,context],[tool])
        if len(calls)>1:
            raise ValueError('Request one run at a time')
        self.history.extend(native)
        if not calls:
            return {'answer':answer}
        call = calls[0]
        if call['name'] != 'workbench_run':
            raise ValueError('Unknown workbench tool')
        args = json.loads(call['arguments'])
        try:
            if args['action']=='layouts':job=self.jobs.layouts(args['count'])
            elif args['action']=='collect':job=self.jobs.collect(args['count'])
            elif args['action']=='train':job=self.jobs.train(args['count'])
            elif args['action']=='cycle':job=self.jobs.cycle(args['count'])
            else:raise ValueError('Unknown run action')
        except (ValueError,OSError) as error:
            self.history.append(model.tool_result(call['id'],{'error':str(error)}))
            raise
        self.history.append(model.tool_result(call['id'],{'id':job['id'],'status':job['status']}))
        return {'answer':answer,'job':job}
