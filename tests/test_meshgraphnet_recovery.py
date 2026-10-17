import copy,random,unittest
import numpy as np
import torch
from geobc.overfit_meshgraphnet import MeshGraphNet,MeshGraphNetConfig
from tools.recover_meshgraphnet_u_overfitting import restore
from tools.run_meshgraphnet_recovery import recovery_clock
from tools.train_sample_overfitting import make_scheduler


class RecoveryTests(unittest.TestCase):
 def test_failed_forward_preserves_next_adam_update_and_rng(self):
  torch.manual_seed(7);c=dict(five=list('abcde'),learning_rate=.001,max_updates_per_arm=5000,lr_schedule=dict(kind='onecycle',pct_start=.1,div_factor=25.,final_div_factor=1000.))
  cfg=MeshGraphNetConfig(latent_size=4,message_passing_steps=1,chunk_size=2);m=MeshGraphNet(cfg).double();o=torch.optim.Adam(m.parameters(),lr=.001);sched=make_scheduler(o,c)
  x=torch.randn(4,28,dtype=torch.double);e=torch.randn(4,4,dtype=torch.double);s=torch.arange(4);r=torch.tensor([1,2,3,0])
  def update(model,opt,sc):
   opt.zero_grad(set_to_none=True);model(x,e,s,r).square().mean().backward();opt.step();sc.step()
  for _ in range(4):update(m,o,sched)
  before=torch.get_rng_state().clone();m(x,e,s,r) # failed attempt stopped immediately after deterministic forward
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
  payload=copy.deepcopy(dict(config=c,step=4,order=c['five'],cursor=4,model=m.state_dict(),optimizer=o.state_dict(),scheduler=sched.state_dict(),arm=dict(partial_backward=True,optimizer_step_in_progress=False),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())))
  n=MeshGraphNet(cfg).double();p=torch.optim.Adam(n.parameters(),lr=.001);ss=make_scheduler(p,c);restore(n,p,ss,payload,c,restore_cuda=False)
  self.assertTrue(torch.equal(before,torch.get_rng_state()));update(m,o,sched);update(n,p,ss)
  self.assertEqual(ss.last_epoch,5)
  for a,b in zip(m.parameters(),n.parameters()):self.assertTrue(torch.equal(a,b))
  self.assertEqual(o.param_groups[0]['lr'],p.param_groups[0]['lr'])
 def test_wait_and_reboot_cannot_extend_original_arm(self):
  old=dict(entry_epoch=1000,entry_monotonic=10,deadline_epoch=65800,deadline_monotonic=64810,boot_id='old');summary=dict(arms={'single_meshgraphnet':dict(elapsed_seconds=dict(UTC=4200,monotonic=3900))})
  first=recovery_clock(old,summary,6000,5000,'old');later=recovery_clock(old,summary,7000,6000,'old');reboot=recovery_clock(old,summary,7000,20,'new')
  self.assertEqual(first['deadline_epoch'],26800);self.assertEqual(first['deadline_monotonic'],25510);self.assertEqual(later['deadline_monotonic'],25510)
  self.assertEqual(reboot['deadline_epoch'],26800);self.assertEqual(reboot['deadline_monotonic']-20,26800-7000)
  with self.assertRaises(TimeoutError):recovery_clock(old,summary,26750,20,'new')


if __name__=='__main__':unittest.main()
