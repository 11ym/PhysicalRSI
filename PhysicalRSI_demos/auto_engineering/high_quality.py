"""Path-traced presentation of saved rigid-body states; never used as task evidence."""
import argparse
import json
from pathlib import Path
import time


def render(args):
    from .harness import verify_episode
    from PhysicalRSI_core.infra.storage import atomic_json,file_digest
    episode=args.episode.resolve();receipt=verify_episode(episode)
    if not receipt['success']:raise ValueError('A success film requires a successful native episode')
    trajectory=json.loads((episode/'trajectory.json').read_text())
    if not trajectory or not trajectory[0].get('render_poses'):raise ValueError('Episode has no recorded render states')
    if args.width<1 or args.height<1 or args.samples<1 or args.count<1:
        raise ValueError('Render dimensions, samples and count must be positive')
    if args.frame is not None and not 0<=args.frame<len(trajectory):
        raise ValueError('Frame index is outside the recorded trajectory')
    output=args.output.resolve()
    source_sha256=file_digest(Path(__file__))
    request={'native_receipt_sha256':file_digest(episode/'result.json'),
        'renderer_source_sha256':source_sha256,'width':args.width,'height':args.height,
        'samples':args.samples,'frame':args.frame,'count':args.count}
    if args.resume:
        if (output/'render.json').exists():raise ValueError('Render is already complete')
        if json.loads((output/'request.json').read_text())!=request:
            raise ValueError('Resume requires identical source, episode and render settings')
    else:
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'request.json',request)
        (output/'renderer-source.py').write_bytes(Path(__file__).read_bytes())
    completed=json.loads((output/'frames.json').read_text()) if args.resume and (output/'frames.json').exists() else {}
    for name,digest in completed.items():
        if file_digest(output/name)!=digest:raise ValueError('Cached frame digest mismatch: '+name)

    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':args.width,'height':args.height,
        'renderer':'PathTracing','samples_per_pixel_per_frame':1,'denoiser':True,
        'max_bounces':16,'max_specular_transmission_bounces':16,'max_volume_bounces':8,
        'multi_gpu':False,'anti_aliasing':0,'disable_viewport_updates':True})
    try:
        import omni.usd
        import omni.replicator.core as rep
        import omni.timeline
        import carb
        import imageio.v2 as imageio
        from PIL import Image
        from pxr import UsdGeom,UsdPhysics,UsdRender,Gf
        context=omni.usd.get_context();context.open_stage(str(episode/'scene.usda'))
        for _ in range(15):app.update()
        stage=context.get_stage()
        omni.timeline.get_timeline_interface().stop()
        # Disable all dynamics: this renderer replays measured state, not a new rollout.
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdRender.Product):prim.SetActive(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):prim.SetActive(False)
        ops={}
        for path in trajectory[0]['render_poses']:
            prim=stage.GetPrimAtPath(path)
            if not prim:raise ValueError('Recorded body absent from source scene: '+path)
            x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.SetResetXformStack(True)
            ops[path]=(x.AddTranslateOp(opSuffix='replay'),x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble,opSuffix='replay'))
        # Give the arm headroom throughout the recorded reach.
        camera=UsdGeom.Camera(stage.GetPrimAtPath('/World/Presentation'))
        camera.GetFocalLengthAttr().Set(330.)
        cx=UsdGeom.Xformable(camera);cx.ClearXformOpOrder()
        cx.AddTransformOp(opSuffix='presentation').Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.25,-1.5,.92),Gf.Vec3d(.31,.10,.28),Gf.Vec3d(0,0,1)).GetInverse())
        rp=rep.create.render_product('/World/Presentation',(args.width,args.height))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(rp)
        settings=carb.settings.get_settings()
        settings.set_bool('/app/asyncRendering',False)
        settings.set_bool('/omni/replicator/asyncRendering',False)
        # Initialize capture cheaply before enabling the final sampling budget.
        rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=True)
        settings.set_int('/rtx/pathtracing/clampSpp',0)
        batch=next(n for n in range(min(16,args.samples),0,-1) if args.samples%n==0)
        settings.set_int('/rtx/pathtracing/spp',batch)
        settings.set_int('/rtx/pathtracing/totalSpp',args.samples)
        settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled',True)
        rows=trajectory if args.frame is None else trajectory[args.frame:args.frame+args.count]
        print('Scene loaded; beginning path-traced capture',flush=True)
        start=time.monotonic()
        writer=None
        if args.frame is None:
            writer=imageio.get_writer(output/'success-4k.mp4',fps=20,codec='libx264',quality=None,
                macro_block_size=8,ffmpeg_params=['-crf','14','-preset','slow','-movflags','+faststart'])
        try:
            for i,row in enumerate(rows):
                frame_path=output/('frame-'+str(row['step'])+'.png')
                if frame_path.name in completed:
                    if writer:writer.append_data(imageio.imread(frame_path))
                    continue
                for path,(position,quaternion) in row['render_poses'].items():
                    translate,orient=ops[path];translate.Set(Gf.Vec3d(*position));orient.Set(Gf.Quatd(quaternion[0],Gf.Vec3d(*quaternion[1:])))
                # Give each capture an explicit, increasing presentation timestamp.
                # All dynamics are disabled; this does not integrate object motion.
                timeline=omni.timeline.get_timeline_interface()
                timeline.set_current_time((i+1)/20);timeline.commit()

                for attempt in range(3):
                    rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=True)
                    frame=rgb.get_data()
                    if frame is not None and frame.size:break
                    print('Waiting for first render-product data',flush=True)
                if frame is None or not frame.size:raise RuntimeError('Path tracer returned no frame')
                transforms=UsdGeom.XformCache()
                for path,(position,quaternion) in row['render_poses'].items():
                    matrix=transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                    if (matrix.ExtractTranslation()-Gf.Vec3d(*position)).GetLength()>1e-6:
                        raise RuntimeError('Presentation changed a recorded body position: '+path)
                    q=matrix.ExtractRotationQuat()
                    dot=q.GetReal()*quaternion[0]+Gf.Dot(q.GetImaginary(),Gf.Vec3d(*quaternion[1:]))
                    if abs(dot)<1-1e-6:raise RuntimeError('Presentation changed a recorded body rotation: '+path)
                frame=frame[...,:3]
                Image.fromarray(frame).save(frame_path)
                completed[frame_path.name]=file_digest(frame_path)
                atomic_json(output/'frames.json',completed)
                if writer:writer.append_data(frame)
                atomic_json(output/'progress.json',{'frames':i+1,'total':len(rows),'seconds':time.monotonic()-start})
                print(json.dumps({'frame':i+1,'total':len(rows),'seconds':round(time.monotonic()-start,2)}),flush=True)
        finally:
            if writer:writer.close()
        atomic_json(output/'render.json',{'state':'completed','kind':'presentation_rerender',
            'renderer_source_sha256':source_sha256,
            'native_receipt_sha256':file_digest(episode/'result.json'),
            'trajectory_sha256':file_digest(episode/'trajectory.json'),'source_episode':str(episode),
            'renderer':'Isaac Sim PathTracing','resolution':[args.width,args.height],
            'presentation_camera':{'eye':[1.25,-1.5,.92],'target':[.31,.10,.28],'focal_length_usd':330.},
            'samples_per_pixel':args.samples,'max_bounces':16,'denoiser':True,'fps':20,
            'frames':len(rows),'physics_rerun':False,'new_success_evidence':False,
            'elapsed_seconds':time.monotonic()-start,
            'actual_settings':{key:settings.get(key) for key in ('/rtx/rendermode','/rtx/pathtracing/spp','/rtx/pathtracing/totalSpp','/rtx/pathtracing/clampSpp','/rtx/pathtracing/maxBounces','/rtx/pathtracing/optixDenoiser/enabled')},
            'files':{p.name:file_digest(p) for p in output.iterdir() if p.is_file()}})
    except Exception:
        import traceback,sys
        traceback.print_exc();sys.stderr.flush();raise
    finally:
        # Avoid expensive presentation sampling during Kit shutdown.
        import carb
        carb.settings.get_settings().set_int('/rtx/pathtracing/spp',1)
        carb.settings.get_settings().set_int('/rtx/pathtracing/totalSpp',1)
        app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=3840);p.add_argument('--height',type=int,default=2160)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--frame',type=int);p.add_argument('--count',type=int,default=1);p.add_argument('--resume',action='store_true')
    render(p.parse_args())
