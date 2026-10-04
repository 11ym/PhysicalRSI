"""Resolve installed demonstration media without copying large assets."""
import json
import os
from pathlib import Path


def catalog():
    data = Path(os.environ.get('PHYSICALRSI_DATA_ROOT', str(Path.home()/'.local/share/physicalrsi')))
    piano = Path(os.environ['ROBOPIANIST_ROOT']) if os.environ.get('ROBOPIANIST_ROOT') else None
    items = []
    media = Path(__file__).resolve().parents[1]/'media'
    for entry in json.loads((media/'manifest.json').read_text())['items']:
        video = media/entry['file']
        if video.is_file():
            items.append({**entry, 'path':str(video.resolve()), 'media':'/media/'+entry['id']})
    def add(key, title, group, video, stages, note, metadata=None):
        if not video.is_file():
            return
        item = dict(id=key, title=title, group=group, path=str(video.resolve()),
                    stages=stages, note=note, media='/media/'+key, mode='Recorded rollout')
        if metadata and metadata.is_file():
            record=json.loads(metadata.read_text())
            item['metrics']={k:record[k] for k in ('f1','precision','recall','audio_source','qualification') if k in record}
        items.append(item)
        return item
    october = next((item for item in items if item['id']=='piano-october'), None)
    iteration = piano/'artifacts/contact-consistency/october-priority-memory/accepted-iteration.json' if piano else None
    if october and iteration and iteration.is_file():
        record = json.loads(iteration.read_text())
        october['evolution'] = {
            'round': record['round'], 'selected': record['selected'],
            'memory_revision': record['memory_revision'],
            'skills': record['skills'],
            'scope': 'Archived contact-skill iteration; separate from the performance recording',
            'candidates': [{
                'name': row['candidate'], 'promoted': row['promoted'],
                'mechanical_f1': row['reports']['october']['mechanical_f1'],
                'contact_f1': row['reports']['october']['contact_supported_f1'],
            } for row in record['outcomes'] if 'october' in row['reports']],
        }
    cycle_media = data/'physicalrsi-demo-workbench/pi05-cycle'
    for manifest in sorted(cycle_media.glob('round-*/successful-demonstrations.json')):
        episodes = json.loads(manifest.read_text())['episodes']
        featured = episodes[:1] + [row for row in episodes if '/repair-' in row['directory']] + episodes[-1:]
        for episode in {row['seed']: row for row in featured}.values():
            relative = episode['directory'].split('/'+manifest.parent.name+'/', 1)[-1]
            video = manifest.parent/relative/'rollout.mp4'
            add('cycle-'+manifest.parent.name+'-'+str(episode['seed']),
                'Mouse · layout '+str(episode['seed']), 'dexjoco', video,
                ['Observe', 'Grasp', 'Place', 'Press', 'Demonstration'],
                'Successful demonstration · '+manifest.parent.name.replace('-', ' '))
    for name,title,stages in [
        ('water_plant','Water a plant',['Locate','Functional grasp','Finger actuation','Release']),
        ('click_mouse','Click a mouse',['Locate','Place','Press','Release']),
        ('hammer_nail','Hammer a nail',['Grasp','Align','Strike','Recover'])]:
        if not any(row['id']=='dex-'+name for row in items):
            add('dex-'+name,title,'dexjoco',data/'dexjoco/0912demo/videos'/f'{name}.mp4',stages,
                'Dexjoco · recorded simulation')
    baseline=data/'physicalrsi-robodojo-1.0-full54-20261003/seeds/seed1/eval_result/RoboDojo'
    for name,title in [('deposit_coin','Deposit a coin'),('push_T','Planar push'),('pour_liquid_into_cup','Pour into a cup')]:
        folder=baseline/name
        if folder.is_dir():
            video=next(folder.glob('**/*cam_head_success.mp4'),None)
            if video is None:video=next(folder.glob('**/*cam_head*.mp4'),None)
            if video:
                add('baseline-'+name,title,'baseline',video,
                    ['Task memory','Skill combination','Observe','Act','Evaluate'],
                    'RoboDojo · archived rollout; not a new evaluation')
    runs = Path(os.environ.get('PHYSICALRSI_SHOWCASE_OUTPUT', str(data/'physicalrsi-demo-workbench/runs')))
    if runs.is_dir():
        for metadata in sorted(runs.glob('*/run.json')):
            record = json.loads(metadata.read_text())
            videos = sorted(metadata.parent.rglob('*.mp4'))
            head_videos = [video for video in videos if 'cam_head' in video.name]
            for index, video in enumerate(head_videos or videos):
                group = record.get('group', 'baseline')
                stages = (['Observe', 'Demonstrate', 'Validate', 'Action chunks'] if group == 'dexjoco'
                          else ['Task memory', 'Skill combination', 'Observe', 'Act', 'Evaluate'])
                row = add('run-'+metadata.parent.name+'-'+str(index), record['title'], group, video,
                    stages, 'Recorded run · '+record['status'])
                if row:row['task'] = record['task']
    return items
