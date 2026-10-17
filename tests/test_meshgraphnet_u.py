import copy,unittest
import numpy as np
import torch
from geobc.meshgraph_topology import quadratic_mesh_edges
from geobc.overfit_meshgraphnet import MeshGraphNet,MeshGraphNetConfig

class MeshGraphTests(unittest.TestCase):
 def test_quadratic_edges_keep_midside_nodes_and_deduplicate(self):
  edge,stats=quadratic_mesh_edges(np.array([range(10),range(10)]),10)
  self.assertEqual(stats['undirected_edges'],12);self.assertEqual(edge.shape,(24,2));self.assertEqual(set(edge.ravel()),set(range(10)))
  pairs=set(map(tuple,edge));self.assertIn((0,4),pairs);self.assertIn((4,1),pairs);self.assertNotIn((0,1),pairs)
  self.assertTrue(all((b,a) in pairs for a,b in pairs))
  with self.assertRaisesRegex(ValueError,'isolated'):quadratic_mesh_edges(np.array([range(10)]),11)
 def test_dense_official_equations_match_chunk_checkpoint_gradients(self):
  torch.manual_seed(12);cfg=MeshGraphNetConfig(latent_size=8,message_passing_steps=2,chunk_size=3)
  m=MeshGraphNet(cfg).double().train();r=copy.deepcopy(m);n=torch.randn(7,28,dtype=torch.float64);e=torch.randn(10,4,dtype=torch.float64);s=torch.tensor([0,1,2,3,4,5,6,1,3,5]);t=torch.tensor([1,2,3,4,5,6,0,0,2,4])
  nodes=r.node_encoder(n);edges=r.edge_encoder(e)
  for block in r.processors:
   de=block.edge_fn(torch.cat((nodes[s],nodes[t],edges),-1));agg=torch.zeros_like(nodes).index_add_(0,t,de);dn=block.node_fn(torch.cat((nodes,agg),-1));nodes=nodes+dn;edges=edges+de
  expected=r.decoder(nodes);actual=m(n,e,s,t);self.assertTrue(torch.allclose(actual,expected,atol=1e-11,rtol=1e-10));expected.square().mean().backward();actual.square().mean().backward()
  for a,b in zip(m.parameters(),r.parameters()):
   self.assertIsNotNone(a.grad);self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-10,rtol=1e-9))
 def test_capacity_and_independent_processors(self):
  m=MeshGraphNet();count=sum(p.numel() for p in m.parameters());self.assertEqual(count,4603863)
  self.assertEqual(len(m.processors),15);self.assertIsNot(m.processors[0].edge_fn.net[0].weight,m.processors[1].edge_fn.net[0].weight)

if __name__=='__main__':unittest.main()
