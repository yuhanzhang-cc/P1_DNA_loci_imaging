"""Post-blind native-pixel montages, with fixed selection and shared windows."""
from pathlib import Path
import random
import numpy as np
from diagnostics.handoff_v2.common import read,table,write,csv_write,fresh,sha


def montage(cases,images,truth,predictions,output,engineering_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=fresh(output);methods=['raw','reference','G0','H0','S0','P0','Spotiflow_F'];source_rows=[]
    with plt.rc_context({'font.size':6,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(len(cases),len(methods),figsize=(7.1,max(2,1.15*len(cases))),squeeze=False,layout='constrained')
        for row,c in enumerate(cases):
            sid=c['sample_id'];image=np.asarray(images[sid]);lo,hi=np.percentile(image,[1,99.8]);y0,x0,y1,x1=[int(c[k]) for k in ('crop_y0','crop_x0','crop_y1','crop_x1')]
            if y1-y0!=64 or x1-x0!=64:raise ValueError('Expected native 64px review patch')
            # Preserve an unmodified native crop and origin for source-data audit.
            crop=image[max(0,y0):min(image.shape[0],y1),max(0,x0):min(image.shape[1],x1)].copy()
            np.savez_compressed(output/(c['anonymous_id']+'_raw.private.npz'),image=crop,origin_yx=[max(0,y0),max(0,x0)],display_window=[lo,hi])
            for col,m in enumerate(methods):
                ax=axes[row,col];ax.imshow(image,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest',origin='upper')
                ax.set(xlim=(x0-.5,x1-.5),ylim=(y1-.5,y0-.5),xticks=[],yticks=[])
                if row==0:ax.set_title(m.replace('Spotiflow_F','Spotiflow transfer'))
                if col==0:ax.set_ylabel(c['anonymous_id']+'\n'+c.get('candidate_is_real_spot',''),fontsize=6)
                points=truth[sid] if m=='reference' else predictions.get(m,{}).get(sid) if m!='raw' else None
                if m not in ('raw','reference') and points is None:ax.text(.5,.5,'pending',ha='center',color='white',transform=ax.transAxes,bbox=dict(facecolor='black',alpha=.6))
                if points is not None:
                    for i,(y,x) in enumerate(np.asarray(points).reshape(-1,2)):
                        if y0<=y<y1 and x0<=x<x1:
                            ax.plot(x,y,'o' if m=='reference' else '+',mfc='none',mec='#56B4E9' if m=='reference' else '#E69F00',ms=4,mew=.7)
                            source_rows.append(dict(anonymous_id=c['anonymous_id'],sample_id=sid,view=m,point_index=i,y=float(y),x=float(x)))
            # Pixel bar, never an invented micrometre calibration.
            axes[row,0].plot([x0+4,x0+12],[y0+57,y0+57],color='white',lw=1.5);axes[row,0].text(x0+4,y0+53,'8 px',color='white',fontsize=5)
        fig.suptitle('SYNTHETIC ENGINEERING ONLY' if engineering_only else 'Post-blind examples; fixed case selection; seed42',fontsize=8)
        for ext in ('svg','pdf','png'):fig.savefig(output/('qualitative_montage.'+ext),dpi=300,facecolor='white')
        plt.close(fig)
    csv_write(output/'overlay_points.private.csv',source_rows,['anonymous_id','sample_id','view','point_index','y','x'])
    write(output/'provenance.json',dict(engineering_only=engineering_only,source_cases=cases,display='whole-native-image percentiles 1 and 99.8, shared across methods; nearest-neighbor display',
        raw_pixels_modified=False,scale_bar='8 native pixels; no physical calibration guessed',method_seed=42,
        interpretation='reference annotations and frozen predictions are separate overlays; human candidate-presence answer is not assignment TP/FP'))
    return dict(output=str(output),cases=len(cases),exports=3,engineering_only=engineering_only)


def build(root,analysis,responses,key,packet,output,dry_run=True):
    if dry_run:return dict(dry_run=True,images_read=0,requires=['completed real blind answers','owner source key','frozen evaluated predictions'],selection='one case per diagnostic stratum, fixed seed20261111',training_started=False)
    from diagnostics.handoff_v2.reviewer import validate_answers
    from diagnostics.mechanism_v1.runner import corpus
    answers=table(responses);sources=table(key);binding=read(packet)
    validate_answers(answers,[r['anonymous_id'] for r in sources],complete=True,expected_packet_id=binding['packet_id'])
    a={r['anonymous_id']:r for r in answers};rng=random.Random(20261111);chosen=[]
    for stratum in sorted({r['stratum'] for r in sources}):
        c=dict(rng.choice(sorted([r for r in sources if r['stratum']==stratum],key=lambda r:r['anonymous_id'])));c['candidate_is_real_spot']=a[c['anonymous_id']]['candidate_is_real_spot'];chosen.append(c)
    data=corpus(root).splits['valid'];images={str(s):x for s,x in zip(data['sample_id'],data['x'])};truth={str(s):x for s,x in zip(data['sample_id'],data['y'])}
    predictions={}
    for method in ('G0','H0','S0','P0','Spotiflow_F'):
        p=Path(analysis)/'evaluated'/method/'seed_42/prediction_points.csv'
        if p.exists():
            predictions[method]={sid:[] for sid in images}
            for r in table(p):predictions[method][r['sample_id']].append([float(r['y']),float(r['x'])])
    result=montage(chosen,images,truth,predictions,output)
    write(Path(output)/'input_binding.json',dict(responses_sha256=sha(responses),source_key_sha256=sha(key),packet_id=binding['packet_id'],selection_seed=20261111,post_blind=True))
    return result
