"""Detect spatially stationary jitter independently of instantaneous velocities."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.ndimage import maximum_filter1d,minimum_filter1d
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(run):
    run=Path(run);r=json.loads((run/'result.json').read_text())
    e=json.loads(run.with_name(run.name+'_execution.json').read_text())
    if not(e['actual_returncode']==0 and e['sources_unchanged'] and r['cleanup_succeeded']
           and e['result_sha256']==sha(run/'result.json') and e['capture_sha256']['physical.npz']==sha(run/'physical.npz')):
        raise ValueError('Normally completed, unchanged physical source required')
    with np.load(run/'physical.npz') as z: xy=z['samples'][...,:2].copy()
    window=150
    if len(xy)<window:raise ValueError('Five actual physical seconds required')
    # Centered filters are used only on complete windows, never padded decisions.
    hi=maximum_filter1d(xy,size=window,axis=0,mode='nearest')
    lo=minimum_filter1d(xy,size=window,axis=0,mode='nearest')
    start=window//2;stop=len(xy)-(window-1-window//2)
    diam=np.linalg.norm((hi-lo)[start:stop],axis=-1)
    rows=[]
    for actor in range(xy.shape[1]):
        mask=diam[:,actor]<.25
        changes=np.diff(np.r_[False,mask,False].astype(int));beg=np.where(changes==1)[0];end=np.where(changes==-1)[0]
        intervals=[{'source_interval':[int(a),int(b+window-1)],'duration_s':float((b-a+window-1)/30)} for a,b in zip(beg,end)]
        longest=max((x['duration_s'] for x in intervals),default=0.)
        rows.append({'actor':actor,'maximum_spatial_stasis_s':longest,'intervals':intervals,
                     'full_window_bbox_diagonal_min_m':float(diam[:,actor].min()),'passes10s_stasis_limit':longest<=10.})
    return {'schema':'crowdsim.actual_spatial_stasis.v3','result_sha256':sha(run/'result.json'),
        'physical_sha256':sha(run/'physical.npz'),'evaluator_sha256':sha(__file__),
        'definition':'Consecutive complete5s root-position windows with bounding-box diagonal below.25m; union duration includes window width. No velocity-based selection.',
        'maximum_allowed_spatial_stasis_s':10.,'normal_turns_or_short_yields_permitted':True,'per_actor':rows,
        'zero_spatial_stasis_over20s':all(x['maximum_spatial_stasis_s']<=20 for x in rows),'zero_spatial_stasis_over10s':all(x['passes10s_stasis_limit'] for x in rows),
        'scope':'Supplementary operational gate for continuous routes without declared long dwell; not a paper benchmark.'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    x=evaluate(a.run);a.output.write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps({'zero_spatial_stasis_over20s':x['zero_spatial_stasis_over20s'],'zero_spatial_stasis_over10s':x['zero_spatial_stasis_over10s'],'per_actor':x['per_actor']}))
    raise SystemExit(0 if x['zero_spatial_stasis_over10s'] else 1)
