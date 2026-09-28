"""Change only MIP Pc to the archived zero-before-breakthrough convention.

No MIP or flow solve is performed. MIP saturation knots and both Kr columns
are preserved bitwise; existing post-breakthrough Pc values are preserved.
"""
import hashlib
import json
import shutil
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
IDENTITY=('fault/region_matrix','window_slice/saturation_region',
          'window_slice/selected_sample_index','window_slice/exact_replay_seed',
          'window_slice/local_perm_m2','window_slice/poro','window_slice/slice_indices')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind=='O' or b.dtype.kind=='O':
        return all(equal(x,y) for x,y in zip(a.flat,b.flat))
    return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else np.array_equal(a,b)

def tables(f):
    g=f['fault/fluid_tables/SGOF']
    ids=g['regions'][()].flatten().astype(int)
    refs=g['tables'][()].flatten()
    assert len(set(ids))==len(ids)==522
    return {int(i):(f[r].name,f[r][()].T) for i,r in zip(ids,refs)}

def same_curve(a,b,col):
    x=np.union1d(a[:,0],b[:,0])
    return float(np.max(np.abs(np.interp(x,a[:,0],a[:,col])-np.interp(x,b[:,0],b[:,col]))))

def main():
    records=json.loads((ROOT/'sources.json').read_text(encoding='utf-8-sig'))
    summary=[]
    for r in records:
        name=r['reference']['name'];src=ROOT/'reference_inputs'/name
        dst=ROOT/'inputs'/name;dst.mkdir(parents=True,exist_ok=True)
        output=dst/'candidate_specific.mat'
        assert not (ROOT/'submissions.json').exists(),'Do not modify submitted inputs'
        if output.exists():
            audit=dst/'pc_only_audit.json'
            if audit.exists():
                checked=json.loads(audit.read_text());assert checked['output_sha256']==sha(output)
                summary.append(dict(name=name,specific=str(output),sha256=sha(output),cores=522,
                    changed_pc_nodes=sum(v['changed_pc_nodes'] for v in checked['records'])))
                continue
            # Keep an unvalidated draft for diagnosis, never replace a qualified input.
            backup=dst/'candidate_specific.unvalidated.latest.mat'
            assert not backup.exists()
            output.rename(backup)
        for kind,pin in r['inputs'].items():assert sha(src/(kind+'.mat'))==pin['sha256']
        shutil.copy2(src/'mip.mat',output)
        changed=[];details=[]
        with h5py.File(src/'mip.mat','r') as h,h5py.File(src/'physical.mat','r') as p,h5py.File(src/'zero.mat','r') as z,h5py.File(output,'r+') as out:
            ht,pt,zt=tables(h),tables(p),tables(z)
            assert ht.keys()==pt.keys()==zt.keys()
            for key in IDENTITY:
                assert np.array_equal(h[key][()],p[key][()]),key
                assert np.array_equal(p[key][()],z[key][()]),key
            for region,(path,a) in ht.items():
                b=pt[region][1];c=zt[region][1]
                assert np.array_equal(b[:,:3],c[:,:3])
                assert same_curve(a,b,3)<1e-7,(name,region,'physical Pc mismatch')
                assert np.isin(c[:,0],a[:,0]).all(),(name,region,'donor knot missing')
                different=np.flatnonzero(b[:,3]!=c[:,3])
                assert len(different)>0 and different[0]==0
                last=int(different[-1]);assert last+1<len(c)
                onset_sg=float(c[last+1,0])
                new=a.copy();mask=a[:,0]<onset_sg
                new[mask,3]=np.interp(a[mask,0],c[:,0],c[:,3])
                assert np.array_equal(new[:,:3],a[:,:3])
                assert np.array_equal(new[~mask,3],a[~mask,3])
                assert same_curve(new,c,3)<1e-7
                assert np.all(np.diff(new[:,3])>=0) and new[0,3]>0
                assert np.isfinite(new).all()
                out[path][...]=new.T
                changed.append(path)
                details.append(dict(region=region,changed_pc_nodes=int(np.sum(new[:,3]!=a[:,3])),
                    old_entry_pa=float(a[0,3]),breakthrough_entry_pa=float(new[0,3]),
                    first_unchanged_postbreakthrough_sg=onset_sg,
                    physical_pc_max_difference_pa=same_curve(a,b,3),
                    zero_pc_max_difference_pa=same_curve(new,c,3)))
            # Keep scientific arrays intact and update only the Pc-treatment label.
            label='metadata/pc_curve_treatment'
            ds=out[label];attrs=dict(ds.attrs)
            text='zero_before_breakthrough_pc_only; archived Pb onset; MIP Kr unchanged'
            del out[label]
            ds=out.create_dataset(label,data=np.array([ord(x) for x in text],dtype=np.uint16).reshape(-1,1))
            for k,v in attrs.items():ds.attrs[k]=v
        with h5py.File(src/'mip.mat','r') as h,h5py.File(output,'r') as out:
            def check(path,obj):
                other=out[path]
                assert set(obj.attrs)==set(other.attrs)
                assert all(equal(obj.attrs[k],other.attrs[k]) for k in obj.attrs)
                if not isinstance(obj,h5py.Dataset):return
                if path=='metadata/pc_curve_treatment':return
                if '/'+path in changed:
                    assert np.array_equal(obj[()][:3],other[()][:3]);return
                if h5py.check_dtype(ref=obj.dtype):
                    assert [h[v].name if v else None for v in obj[()].flat]==[out[v].name if v else None for v in other[()].flat]
                else:assert equal(obj[()],other[()]),path
            h.visititems(check)
        audit=dict(passed=True,name=name,source=r,output_sha256=sha(output),
            all_522_kr_and_saturation_knots_bitwise_equal=True,
            postbreakthrough_pc_nodes_bitwise_equal=True,
            donor_physical_pc_functions_match=True,donor_zero_pc_functions_match=True,
            all_other_scientific_datasets_exact=True,
            treatment='Archived exact breakthrough pressure at Sg=0; linear onset to first sampled post-breakthrough state.',
            caveat='Pc table convention only; no new phase-entry switch or saturation clamp in reservoir simulator.',
            records=details)
        (dst/'pc_only_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        summary.append(dict(name=name,specific=str(output),sha256=sha(output),cores=522,
                            changed_pc_nodes=sum(v['changed_pc_nodes'] for v in details)))
        print(summary[-1],flush=True)
    (ROOT/'input_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
