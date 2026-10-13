"""Plot saved full-node final predictions and errors on the original surface."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from geobc.physics_inputs import local_path
from geobc.final_fields import read_final_fields,LOADS


def surface(cells):
    # Keep only exterior corner faces, then retain their actual TET10 midnodes.
    patterns=np.array([[0,1,2,4,5,6],[0,3,1,7,8,4],[1,3,2,8,9,5],[0,2,3,6,9,7]])
    faces=cells[:,patterns].reshape(-1,6)
    _,inverse,counts=np.unique(np.sort(faces[:,:3],axis=1),axis=0,return_inverse=True,return_counts=True)
    exterior=faces[counts[inverse]==1]
    parts=np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
    return exterior[:,parts].reshape(-1,3)


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(Path(args.source).read_text())
    state=json.loads((Path(args.run)/'summary.json').read_text())
    if state['status']!='completed' or not state['prediction_barrier_completed']:raise ValueError('Completed frozen final predictions required')
    record=catalog['truth_sources'][args.cad]
    with np.load(local_path(record['GRID_cache']),allow_pickle=False) as source:
        vertices,cells,GRID=source['vertices_mm'],source['cells10'],source['grid_node_ids']
    if vertices.dtype!=np.float64 or cells.dtype!=np.int64 or not np.isfinite(vertices).all():raise ValueError('Geometry schema differs')
    truths=read_final_fields(local_path(record['CSV']),np.load(local_path(record['permutation']),allow_pickle=False),GRID,True)
    triangles=surface(cells)
    # Same orthographic projection convention as the reference PPT field panels.
    azimuth,elevation=np.deg2rad([-65,24])
    right=np.array([-np.sin(azimuth),np.cos(azimuth),0.])
    up=np.array([-np.sin(elevation)*np.cos(azimuth),-np.sin(elevation)*np.sin(azimuth),np.cos(elevation)])
    toward=np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
    centered=vertices-(vertices.min(0)+vertices.max(0))/2
    xy=np.column_stack((centered@right,centered@up));depth=centered@toward
    order=np.argsort(depth[triangles].mean(1),kind='stable')
    geometry=xy[triangles][order];lo=xy.min(0);hi=xy.max(0);margin=(hi-lo)*.07
    font=Path('/mnt/c/Windows/Fonts/arial.ttf')
    if font.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
        bold_font=font.with_name('arialbd.ttf')
        if bold_font.exists():font_manager.fontManager.addfont(str(bold_font))
    plt.rcParams['axes.unicode_minus']=False
    all_metrics={};loads=LOADS if args.load=='all' else [args.load]
    for load in loads:
        truth=truths[load].astype(np.float64)
        modes=['zero_seed'+str(args.seed),'fieldwise_seed'+str(args.seed)]
        predictions=[np.load(local_path(state['predictions'][mode][args.cad+'/'+load]),allow_pickle=False) for mode in modes]
        if any(p.shape!=truth.shape or p.dtype!=np.float64 or not np.isfinite(p).all() for p in predictions):raise ValueError('Complete saved field predictions required')
        values=[truth]+predictions
        magnitude=[np.linalg.norm(v[:,:3],axis=1) for v in values]
        stress=[v[:,3] for v in values]
        Uerrors=[np.linalg.norm(p[:,:3]-truth[:,:3],axis=1) for p in predictions]
        Serrors=[np.abs(p[:,3]-truth[:,3]) for p in predictions]
        metrics={mode:dict(U_relative_percent=float(np.linalg.norm(p[:,:3]-truth[:,:3])/np.linalg.norm(truth[:,:3]))*100,S_relative_percent=float(np.linalg.norm(p[:,3]-truth[:,3])/np.linalg.norm(truth[:,3]))*100) for mode,p in zip(modes,predictions)}
        all_metrics[load]=metrics
        Unorm=Normalize(0,max(v.max() for v in magnitude))
        Smax=max(np.abs(v).max() for v in stress)
        Snorm=Normalize(-Smax,Smax)
        load_name=dict(ver='vertical',hor='horizontal',tor='torsion')[load]
        book=PdfPages(output/(load+'_comparison.pdf'))
        for field,scalar,residuals,norm,unit in [('displacement',magnitude,Uerrors,Unorm,'mm'),('stress',stress,Serrors,Snorm,'MPa')]:
            RMS=np.sqrt(np.mean(np.sum(truth[:,:3]**2,axis=1))) if field=='displacement' else np.sqrt(np.mean(truth[:,3]**2))
            normalized_errors=[e/RMS*100 for e in residuals]
            error_norm=Normalize(0,50,clip=True)
            fig=plt.figure(figsize=(16,9),facecolor='white')
            fig.text(.0525,.932,field.capitalize()+' fields: prediction and error',fontsize=30.5,fontweight='bold',color='#172638',va='center')
            fig.text(.055,.862,f'CAD {args.cad} / {load_name}  |  Registered seed {args.seed}  |  Saved final predictions',fontsize=14,color='#657383')
            for row,model_name in enumerate(['ZERO baseline','Shared-trunk model']):
                metric=metrics[modes[row]]['U_relative_percent' if field=='displacement' else 'S_relative_percent']
                model_y=.787-row*.35
                fig.text(.055,model_y,f'{model_name}  |  {metric:.2f}% '+('displacement' if field=='displacement' else 'stress')+' L2',fontsize=20,fontweight='bold',color='#ed3a94')
                base=.478-row*.35
                for col,(label,values_here) in enumerate(zip(['FEM truth','Prediction','Absolute error / true RMS'],[scalar[0],scalar[row+1],normalized_errors[row]])):
                    left=.048+col*.321
                    ax=fig.add_axes([left,base,.262,.245])
                    current_norm=error_norm if col==2 else norm
                    cmap=plt.get_cmap('magma' if col==2 else 'turbo')
                    face_values=values_here[triangles].mean(1)[order]
                    ax.add_collection(PolyCollection(geometry,facecolors=cmap(current_norm(face_values)),edgecolors='none',rasterized=True))
                    ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1]);ax.set_aspect('equal');ax.set_axis_off()
                    fig.text(left+.14,base+.271,label,ha='center',fontsize=13.5,color='#111111')
                    cax=fig.add_axes([left+.272,base+.015,.009,.22])
                    cb=fig.colorbar(ScalarMappable(norm=current_norm,cmap=cmap),cax=cax,extend='max' if col==2 else 'both' if field=='stress' else 'neither')
                    cb.ax.tick_params(labelsize=9,length=2);cb.outline.set_linewidth(.5)
                    cb.set_label('%' if col==2 else unit,fontsize=10,labelpad=3)
            fig.text(.055,.061,'Original, undeformed geometry. Same field scale and camera for both models; all-node L2 errors.',fontsize=11,color='#657383')
            fig.text(.055,.027,'Error colors: 0–50% of true RMS, larger values saturated, as in the reference PPT.',fontsize=10,color='#657383')
            fig.savefig(output/(load+'_'+field+'.png'),dpi=200)
            book.savefig(fig,dpi=200)
            plt.close(fig)
        book.close()
        # A single overview keeps the two PPT-style layouts intact, without blending.
        from PIL import Image
        with Image.open(output/(load+'_displacement.png')) as top,Image.open(output/(load+'_stress.png')) as bottom:
            combined=Image.new('RGB',(top.width,top.height+bottom.height),'white');combined.paste(top,(0,0));combined.paste(bottom,(0,top.height));combined.save(output/(load+'_comparison.png'))
    metadata=dict(CAD=args.cad,seed=args.seed,loads=loads,nodes=len(vertices),cells=len(cells),surface_subtriangles=len(triangles),models=modes,full_node_metrics=all_metrics,stress_label='Original signed scalar S in MPa; no inferred VM sign rule',visualization='Reference PPT three-column layout, turbo fields, magma error; linear field scales; errors normalized by all-node truth RMS, common display limit 50%',template_slide='p9',template_presentation='1g5S0QCZSDR_igYFWFe20SQ3nw2yvgYbXk_t4QSmKpBU',truth_use='Read the revealed CAD fields for requested visualization after the completed final prediction barrier; no training or new inference')
    (output/'figure_metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'README.md').write_text('# Saved-model field comparison\n\nFixed first final CAD '+args.cad+', registered seed '+str(args.seed)+'. The reference Google Slides field-page layout and palette are followed: FEM truth, prediction and error columns; white background, Arial text, navy title, pink L2 label, turbo field maps and magma error maps with vertical bars. Two rows show ZERO and FIELDWISE. Separate displacement and stress figures preserve the reference 16:9 slide format.\n\nDisplacement error is the vector-difference norm divided by full-node true vector RMS; stress error is absolute signed-scalar difference divided by full-node true stress RMS. Both are shown in percent on a common 0–50% scale, with larger errors saturated, as in the reference figure. Full-node L2 metrics are never clipped. Field scales are shared between truth and both models; original boundary midnodes are retained, without enlarged deformation.\n\nAll three canonical loads have PNG and PDF figures, plus combined PNG overviews. No model inference, training or final-set tuning was performed. The revealed labels were read solely for the requested visualization.\n')
    table='\n| Load | Displacement | Stress | Two-page PDF |\n|---|---|---|---|\n'
    for load in loads:table+=f'| {load} | [PNG]({load}_displacement.png) | [PNG]({load}_stress.png) | [PDF]({load}_comparison.pdf) |\n'
    with (output/'README.md').open('a') as file:file.write(table)
    print(json.dumps(metadata))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',default='runs/final_model_evaluation');p.add_argument('--cad',default='21_119');p.add_argument('--seed',type=int,default=2026);p.add_argument('--load',choices=['all']+LOADS,default='all');p.add_argument('--output',default='results/field_comparison')
    run(p.parse_args())
