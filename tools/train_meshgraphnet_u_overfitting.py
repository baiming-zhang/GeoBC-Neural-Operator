"""Guarded full-mesh static-displacement MeshGraphNet fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tools import train_sample_overfitting as trainer
from geobc.overfit_models import relative_loss
from geobc.overfit_meshgraphnet import MeshGraphNet,MeshGraphNetConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True);args=p.parse_args()
 sources=json.loads(Path(args.source).read_text());graph_stats={};output=Path(args.output)
 def prepare_graph(case,resolution=64):
  key=case['key'];indices=np.load(sources['graphs'][key],allow_pickle=False)
  if indices.ndim!=2 or indices.shape[1]!=2 or indices.min()<0 or indices.max()>=len(case['xyz']):raise ValueError('Aligned original mesh graph required')
  send=torch.as_tensor(indices[:,0].copy(),dtype=torch.long,device='cuda');recv=torch.as_tensor(indices[:,1].copy(),dtype=torch.long,device='cuda')
  relative=case['xyz'][send]-case['xyz'][recv]
  case['senders']=send;case['receivers']=recv;case['edge_features']=torch.cat((relative,torch.linalg.vector_norm(relative,dim=-1,keepdim=True)),dim=-1)
  graph_stats[key]=dict(nodes=len(case['xyz']),directed_edges=len(send),edge_channels=4)
  trainer.write(output/'graph_cache.json',dict(graph_reads=len(graph_stats),graphs=graph_stats,topology_source='existing original TETRA10 geometry cache',graph_target_reads=0,stress_reads=0,heldout_reads=0))
 def make(method,seed,settings=None,out_channels=3):
  if method!='meshgraphnet' or out_channels!=3:raise ValueError('Only static U3 MeshGraphNet belongs to this stage')
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  return MeshGraphNet(MeshGraphNetConfig(output_channels=out_channels,**(settings or {}))).cuda()
 def backward(method,model,case,guard):
  guard();pred=model(case['inputs'],case['edge_features'],case['senders'],case['receivers']);guard()
  loss=relative_loss(pred,case['target'],case['energy'],len(case['xyz']))
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-graph loss')
  loss.backward();guard();return float(loss.detach())
 @torch.no_grad()
 def predict(method,model,case,guard):
  guard();pred=model(case['inputs'],case['edge_features'],case['senders'],case['receivers']);guard();return pred.float()
 trainer.prepare_grid=prepare_graph;trainer.make_model=make;trainer.backward_case=backward;trainer.predict_case=predict
 trainer.run(args)

if __name__=='__main__':main()
