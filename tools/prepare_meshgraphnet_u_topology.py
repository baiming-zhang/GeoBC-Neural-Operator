"""Build graph connectivity from existing TRAIN H5 geometry only, never nodal fields."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np
from geobc.meshgraph_topology import quadratic_mesh_edges

def run(workspace,source_path,output):
 start=time.time();sources=json.loads(source_path.read_text());repo=Path.cwd();cfg=json.loads((repo/'configs/fifty_u_overfitting.json').read_text());expected={r['sample_key']:r['nodes'] for r in json.loads((repo/'configs/fifty_u_samples.json').read_text())['cases']}
 output.mkdir(parents=True,exist_ok=False);records=[];sources['graphs']={};progress=output/'status.json'
 for key in cfg['fifty']:
  with h5py.File(sources['mesh_h5'][key],'r') as f:vertices=np.array(f['vertices'],dtype=np.float32);cells=np.array(f['cells'],dtype=np.int64)
  src=sources['cases'][key]
  if isinstance(src,str):
   with np.load(src,allow_pickle=False) as f:xyz=np.array(f['xyz'],dtype=np.float32)
  else:xyz=np.array(np.load(src['xyz_path'],mmap_mode='r',allow_pickle=False),dtype=np.float32)
  norm=(vertices-vertices.min(0))/np.ptp(vertices,axis=0).max()
  if len(xyz)!=expected[key] or xyz.shape!=vertices.shape or not np.allclose(norm,xyz,rtol=0,atol=2e-7):raise ValueError('TRAIN input/H5 node order or coordinates mismatch: '+key)
  edges,stats=quadratic_mesh_edges(cells,len(xyz));path=output/(key.split('/')[0]+'_edges.npy');np.save(path,edges,allow_pickle=False);sources['graphs'][key]=str(path)
  records.append(dict(sample_key=key,**stats,coordinate_alignment_max_error=float(np.abs(norm-xyz).max()),source_H5_members=['vertices','cells'],source_H5_name=Path(sources['mesh_h5'][key]).name))
  progress.write_text(json.dumps(dict(status='building',complete_graphs=len(records),elapsed_seconds=time.time()-start,U_label_reads=0,stress_reads=0,heldout_reads=0),indent=2)+'\n')
  print(json.dumps(dict(case=key,**stats)),flush=True)
 manifest=dict(status='completed',cases=records,graphs=50,geometry_H5_reads=50,aligned_XYZ_reads=50,U_label_reads=0,stress_reads=0,heldout_reads=0,elapsed_seconds=time.time()-start,topology='original TETRA10 corner-midpoint-corner edge segments, both directions, deduplicated; every original node retained')
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');progress.write_text(json.dumps(manifest,indent=2)+'\n')
 (repo/'configs/meshgraphnet_graphs.json').write_text(json.dumps(manifest,indent=2)+'\n');source_path.write_text(json.dumps(sources,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.workspace,a.source,a.output)
