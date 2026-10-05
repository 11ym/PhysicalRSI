"""A self-contained local evidence viewer; all scores come from native receipts."""
import html
import json
from pathlib import Path
from PhysicalRSI_core.infra.storage import read_json
from .harness import verify_episode


def render(workspace):
    root=Path(workspace);round_root=root/'round-001';cards=[]
    for p in [round_root/'development',*sorted((round_root/'episodes').glob('*/*'))]:
        if not (p/'result.json').exists():continue
        d=verify_episode(p);relative=p.relative_to(root);passed=d['success']
        checks=' · '.join(k.replace('_',' ') for k,v in d['checks'].items() if not v) or 'All declared checks passed'
        cards.append(f'''<article><div class="cardhead"><span>{html.escape(str(relative))}</span><b class="{'pass' if passed else 'fail'}">{'PASS' if passed else 'FAIL'}</b></div>
<video controls muted preload="metadata" poster="{relative}/final.png" src="{relative}/episode.mp4"></video>
<p>{html.escape(checks)}</p><small>Seed {d['seed']} · transit {d['memory']['transit_height_m']:.3f} m · {d['physics_steps']} physics steps</small>
<p><a href="{relative}/result.json">Receipt</a> <a href="{relative}/public-observation.json">Public perception</a> <a href="{relative}/trajectory.json">Joint trace</a></p></article>''')
    summary=read_json(root/'result.json');diagnosis=read_json(round_root/'diagnosis.json') if (round_root/'diagnosis.json').exists() else {}
    selection=read_json(round_root/'selection.json') if (round_root/'selection.json').exists() else {}
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>physicalRSI · Lab engineering</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f0e9;color:#142b30;font:16px/1.55 system-ui}main{max-width:1280px;margin:auto;padding:48px 32px}header{border-top:4px solid #183c40;padding-top:18px}label{font-size:12px;letter-spacing:.16em;text-transform:uppercase}h1{font:clamp(38px,6vw,76px)/1.08 Georgia,serif;margin:22px 0}header p{max-width:750px}.steps{display:flex;gap:12px;flex-wrap:wrap;margin:32px 0}.steps span{border:1px solid #9caaa5;padding:10px 16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:24px}article{background:#fff;padding:18px;border:1px solid #d8ddd7}.cardhead{display:flex;justify-content:space-between;font-size:12px;gap:12px;margin-bottom:12px}video{width:100%;background:#142b30}.pass{color:#167353}.fail{color:#ac402c}small{color:#596965}a{color:#225f65;margin-right:12px}details{margin:28px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e7e9e2;padding:20px;font-size:12px}footer{border-top:1px solid #b5bfba;padding-top:20px;font-size:13px;margin-top:40px}</style>
<main><header><label>physicalRSI / Auto engineering / 001</label><h1>A skill becomes<br>an engineering result.</h1><p>A blue sample, a receiving rack, and a beaker in the way. Observe a real simulation failure, test a clearance repair, and retain only the version supported by new paired layouts.</p></header>
<div class="steps"><span>01 · Describe the robot</span><span>02 · Build the lab</span><span>03 · Compose a skill</span><span>04 · Test & retain</span></div>'''
    content+=f'<p>Selected policy: <strong>{html.escape(summary["selected_policy"])}</strong></p><div class="grid">'+''.join(cards)+'</div>'
    for title,value in [('Diagnosis & proposed repair',diagnosis),('Paired selection',selection),('Lineage revision',summary)]:
        content+=f'<details><summary>{title}</summary><pre>{html.escape(json.dumps(value,indent=2))}</pre></details>'
    content+='''<footer>Native Isaac Sim simulation development. Procedurally built scene; color-labelled RGB-D perception. A bounded repair rule, not a general engineering agent. No liquid simulation or physical qualification. Simulator object state is restricted to evaluation and presentation recording; the reviewed policy is not an OS-isolation security claim.<p>Inspired by <a href="https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics">General Robotics: Auto-Engineering</a>.</p></footer></main></html>'''
    (root/'index.html').write_text(content)
    return root/'index.html'
