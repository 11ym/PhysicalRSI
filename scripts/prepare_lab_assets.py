"""Download credited lab assets outside the repository and extract render-only props.

Run with Python + usd-core. These assets are not redistributed with physicalRSI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def prepare(root):
    from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
    import math
    root.mkdir(parents=True,exist_ok=True)
    def fetch(url,path,expected=None):
        if not path.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'physicalRSI-research-demo/1.0'})
            with urllib.request.urlopen(request,timeout=120) as r:path.write_bytes(r.read())
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and sha!=expected:raise ValueError('Unexpected upstream asset content: '+str(path))
        return sha
    source='https://media.githubusercontent.com/media/Rui-li023/LabUtopia/main/assets/chemistry_lab/lab_001/lab_001.usd'
    original=root/'labutopia-lab001.usd'
    sha=fetch(source,original,'26212d40a78cd28f2bc3a38b2e06f05875b3096501c7d6dd5ccbe9a6e9019983')
    stage=Usd.Stage.Open(str(original),load=Usd.Stage.LoadNone)
    cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    records=[]
    for name,prim_path in [('beaker','/World/beaker1'),('drying_oven','/World/DryingBox_02')]:
        original_prim=stage.GetPrimAtPath(prim_path)
        geometry=[]
        skipped=[]
        for prim in Usd.PrimRange(original_prim):
            if not prim.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(prim);points=mesh.GetPointsAttr().Get()
            if not points:continue
            matrix=cache.GetLocalToWorldTransform(prim)
            vertices=[matrix.Transform(Gf.Vec3d(*point)) for point in points]
            if not all(math.isfinite(v) for point in vertices for v in point):
                skipped.append(str(prim.GetPath()));continue
            geometry.append((prim,mesh,vertices))
        if not geometry:raise ValueError('No finite geometry in '+prim_path)
        vertices=[v for _,_,points in geometry for v in points]
        lo=Gf.Vec3d(*[min(v[k] for v in vertices) for k in range(3)])
        hi=Gf.Vec3d(*[max(v[k] for v in vertices) for k in range(3)]);size=hi-lo
        output=root/(name+'.usda')
        target=Usd.Stage.CreateNew(str(output)) if not output.exists() else Usd.Stage.Open(str(output))
        if target.GetPrimAtPath('/Prop'):target.RemovePrim('/Prop')
        asset=UsdGeom.Xform.Define(target,'/Prop');target.SetDefaultPrim(asset.GetPrim());UsdGeom.SetStageUpAxis(target,'Z');UsdGeom.SetStageMetersPerUnit(target,1.)
        count=0
        for prim,mesh,vertices in geometry:
            pts=[((q[0]-lo[0])/size[0]-.5,(q[1]-lo[1])/size[1]-.5,(q[2]-lo[2])/size[2]) for q in vertices]
            if name=='drying_oven':pts=[(-x,-y,z) for x,y,z in pts]
            out=UsdGeom.Mesh.Define(target,'/Prop/Mesh'+str(count));out.CreatePointsAttr(pts)
            out.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get());out.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get());out.CreateSubdivisionSchemeAttr('none');out.CreateDoubleSidedAttr(True)
            mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            label=str(mat.GetPath()).lower() if mat else ''
            if 'panel' in str(prim.GetPath()).lower():label='black'
            elif 'handle' in str(prim.GetPath()).lower():label='steel'
            category='steel' if any(x in label for x in ['steel','stainless','aluminum']) else ('dark' if any(x in label for x in ['black','dark','glass','deepgray']) else 'white')
            out.GetPrim().CreateAttribute('physicalRSI:material',Sdf.ValueTypeNames.String).Set(category)
            count+=1
        target.GetRootLayer().Save()
        records.append({'file':output.name,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_prim':prim_path,'mesh_count':count,'skipped_nonfinite_meshes':skipped})
    url='https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/studio_small_09_2k.hdr'
    hdri=root/'studio_small_09_2k.hdr';hdrisha=fetch(url,hdri)
    credits={'labutopia':{'source':source,'page':'https://github.com/Rui-li023/LabUtopia','license':'CC BY-NC 4.0','source_sha256':sha,'changes':'Selected meshes, normalized geometry, replaced render materials; no source scene or policies imported.','derived_assets':records},'lighting':{'source':url,'page':'https://polyhaven.com/a/studio_small_09','license':'CC0-1.0','sha256':hdrisha}}
    (root/'lab-assets.json').write_text(json.dumps(credits,indent=2)+'\n')
    print(json.dumps(credits,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-dir',type=Path,required=True)
    prepare(p.parse_args().assets_dir.resolve())
