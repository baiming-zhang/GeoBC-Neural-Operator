import copy,json,random,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from tools.train_sample_overfitting import make_scheduler
from tools.recover_fifty_u_overfitting import restore
from tools.queue_fifty_u_recovery import recovery_clock


class RecoveryTests(unittest.TestCase):
    def test_wait_cannot_extend_original_arm_or_whole_deadline(self):
        original=dict(entry_epoch=1000.,entry_monotonic=100.,deadline_epoch=29800.,deadline_monotonic=28900.,boot_id='boot')
        summary={'arms':{'fifty_'+m:dict(status='completed',updates=5000,elapsed_seconds=dict(UTC=100.,monotonic=90.)) for m in ('mlp','ngp4','hashgrid','tcnn_hash','fno','transolver')}}
        clock=recovery_clock(original,summary,3000.,2100.,'boot')
        self.assertEqual(clock['deadline_epoch'],12400.);self.assertEqual(clock['deadline_monotonic'],11440.)
        later=recovery_clock(original,summary,5000.,4100.,'boot');self.assertEqual(later['deadline_epoch'],clock['deadline_epoch'])
        with self.assertRaises(TimeoutError):recovery_clock(original,summary,12399.,11439.,'boot')

    def test_step21_Adam_scheduler_and_RNG_resume_without_replay(self):
        c=json.loads(Path('configs/fifty_u_overfitting.json').read_text())
        p=torch.nn.Linear(1,1);opt=torch.optim.Adam(p.parameters(),lr=.001);sch=make_scheduler(opt,c)
        for i in range(21):opt.zero_grad();p(torch.ones(1,1)).square().sum().backward();opt.step();sch.step()
        payload=copy.deepcopy(dict(config=c,order=c['fifty'],step=21,cursor=21,arm=dict(partial_backward=False,optimizer_step_in_progress=False),model=p.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[])))
        expected=(random.random(),np.random.random(),torch.rand(1));opt.zero_grad();p(torch.ones(1,1)).square().sum().backward();opt.step();sch.step()
        q=torch.nn.Linear(1,1);other=torch.optim.Adam(q.parameters(),lr=.001);schedule=make_scheduler(other,c)
        with patch('torch.cuda.set_rng_state_all') as gpu_rng:restore(q,other,schedule,payload,c);gpu_rng.assert_called_once_with([])
        observed=(random.random(),np.random.random(),torch.rand(1));self.assertEqual(expected[:2],observed[:2]);self.assertTrue(torch.equal(expected[2],observed[2]))
        other.zero_grad();q(torch.ones(1,1)).square().sum().backward();other.step();schedule.step()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(p.parameters(),q.parameters())))
        self.assertEqual(other.param_groups[0]['lr'],opt.param_groups[0]['lr']);self.assertEqual(schedule.last_epoch,22)


if __name__=='__main__':unittest.main()
