"""MeshGraphNet encode/process/decode, adapted to static U3 on full TETRA10 graphs.

Architecture reference: DeepMind meshgraphnets/core_model.py (Apache-2.0).
This PyTorch implementation is not the original TensorFlow runtime or rollout task.
"""
from dataclasses import asdict,dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

SOURCE='https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/core_model.py'

@dataclass(frozen=True)
class MeshGraphNetConfig:
 latent_size:int=180
 message_passing_steps:int=15
 hidden_layers:int=2
 node_channels:int=28
 edge_channels:int=4
 output_channels:int=3
 chunk_size:int=32768
 checkpoint_blocks:bool=True

class MLP(nn.Module):
 def __init__(self,inputs,outputs,hidden,layers=2,layer_norm=True):
  super().__init__();blocks=[]
  for _ in range(layers):blocks.extend((nn.Linear(inputs,hidden),nn.ReLU()));inputs=hidden
  blocks.append(nn.Linear(inputs,outputs))
  if layer_norm:blocks.append(nn.LayerNorm(outputs))
  self.net=nn.Sequential(*blocks)
 def forward(self,x):return self.net(x)

class Processor(nn.Module):
 def __init__(self,cfg):
  super().__init__();h=cfg.latent_size;self.cfg=cfg
  self.edge_fn=MLP(3*h,h,h,cfg.hidden_layers)
  self.node_fn=MLP(2*h,h,h,cfg.hidden_layers)
 def forward(self,nodes,edges,senders,receivers):
  def edge_chunk(n,e,s,r):return self.edge_fn(torch.cat((n[s],n[r],e),-1))
  chunks=[];size=self.cfg.chunk_size
  for start in range(0,len(edges),size):
   args=(nodes,edges[start:start+size],senders[start:start+size],receivers[start:start+size])
   chunks.append(checkpoint(edge_chunk,*args,use_reentrant=False) if self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() else edge_chunk(*args))
  delta_edges=torch.cat(chunks)
  aggregated=torch.zeros_like(nodes).index_add_(0,receivers,delta_edges)
  def node_chunk(n,a):return self.node_fn(torch.cat((n,a),-1))
  chunks=[]
  for start in range(0,len(nodes),size):
   args=(nodes[start:start+size],aggregated[start:start+size])
   chunks.append(checkpoint(node_chunk,*args,use_reentrant=False) if self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() else node_chunk(*args))
  # Official core aggregates the new edge delta before adding edge residuals.
  return nodes+torch.cat(chunks),edges+delta_edges

class MeshGraphNet(nn.Module):
 def __init__(self,config=None):
  super().__init__();self.cfg=config or MeshGraphNetConfig();c=self.cfg;h=c.latent_size
  self.node_encoder=MLP(c.node_channels,h,h,c.hidden_layers)
  self.edge_encoder=MLP(c.edge_channels,h,h,c.hidden_layers)
  self.processors=nn.ModuleList([Processor(c) for _ in range(c.message_passing_steps)])
  self.decoder=MLP(h,c.output_channels,h,c.hidden_layers,layer_norm=False)
  self.source_config={**asdict(c),'source':SOURCE,'implementation':'PyTorch reimplementation of official encoder/processor/decoder; static displacement adaptation',
   'input':'normalizedXYZ+25 original BC/material conditions','edge_features':'normalized mesh-position sender-receiver delta3 and its Euclidean length',
   'topology':'original TETRA10 six quadratic edges split at original midside node; directed, deduplicated, all original nodes',
   'aggregation':'sum of edge delta by receiver before edge residual; independent processor weights',
   'teacher_or_previous_displacement':False,'sample_id':False,'node_sampling':False,'input_running_normalizer':False,'output':'signed Uxyz, no rollout integration'}
 def _chunked(self,fn,x):
  out=[]
  for start in range(0,len(x),self.cfg.chunk_size):
   v=x[start:start+self.cfg.chunk_size]
   out.append(checkpoint(fn,v,use_reentrant=False) if self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() else fn(v))
  return torch.cat(out)
 def forward(self,node_inputs,edge_inputs,senders,receivers):
  nodes=self._chunked(self.node_encoder,node_inputs);edges=self._chunked(self.edge_encoder,edge_inputs)
  for block in self.processors:
   nodes,edges=checkpoint(block,nodes,edges,senders,receivers,use_reentrant=False) if self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() else block(nodes,edges,senders,receivers)
  return self._chunked(self.decoder,nodes)
