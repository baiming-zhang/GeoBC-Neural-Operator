import unittest,copy
import torch
import numpy as np
from geobc.overfit_models import relative_loss
from tools.train_sample_overfitting import errors,read_case_arrays,learning_rate_multiplier,make_scheduler,update_early_stopping


class OverfittingTests(unittest.TestCase):
    def test_chunked_full_node_gradient(self):
        torch.manual_seed(1)
        target=torch.randn(19,4,dtype=torch.float64)
        energy=torch.stack((target[:,:3].square().sum(1).mean(),target[:,3].square().mean()))
        p=torch.randn(19,4,dtype=torch.float64,requires_grad=True)
        whole=relative_loss(p,target,energy,len(p));whole.backward();expected=p.grad.clone();p.grad=None
        split=sum(relative_loss(p[a:b],target[a:b],energy,len(p)) for a,b in [(0,7),(7,19)])
        split.backward()
        self.assertTrue(torch.allclose(whole,split,atol=1e-14,rtol=1e-14))
        self.assertTrue(torch.allclose(p.grad,expected,atol=1e-14,rtol=1e-14))

    def test_signed_vector_metric(self):
        truth=torch.tensor([[1.,-2.,3.,-4.],[-2.,4.,-6.,8.]])
        result=errors(truth,truth*.995)
        self.assertAlmostEqual(result['U_percent'],.5,places=4)
        self.assertAlmostEqual(result['S_percent'],.5,places=4)
        flipped=truth.clone();flipped[:,3]*=-1
        self.assertAlmostEqual(errors(truth,flipped)['S_percent'],200)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):errors(torch.ones(3,4),torch.full((3,4),float('nan')))

    def test_u_only_loss_and_chunked_gradient(self):
        target=torch.tensor([[1.,-2.,3.],[-2.,4.,-6.]],dtype=torch.float64)
        energy=target.square().sum(1).mean()[None]
        pred=(target*.9).clone().requires_grad_()
        whole=relative_loss(pred,target,energy,len(pred));whole.backward();gradient=pred.grad.clone();pred.grad=None
        split=sum(relative_loss(pred[i:i+1],target[i:i+1],energy,len(pred)) for i in range(2));split.backward()
        self.assertAlmostEqual(float(whole.detach()),.01,places=12)
        self.assertTrue(torch.allclose(pred.grad,gradient,atol=1e-14,rtol=1e-14))
        self.assertEqual(set(errors(target,pred.detach())),{'U_percent'})
        self.assertAlmostEqual(errors(target,pred.detach())['U_percent'],10.,places=10)

    def test_u_only_does_not_read_stress(self):
        source={'label_alignment':np.array('official CSV GRID coordinates -> H5 vertices'),'sample_key':np.array('case/ver'),'xyz':np.zeros((2,3)),'features':np.zeros((2,25)),'displacement':np.ones((2,3))}
        xyz,features,u=read_case_arrays(source,'case/ver','u')
        self.assertEqual(u.shape,(2,3))
        self.assertEqual(u.dtype,np.float32)
        with self.assertRaises(KeyError):read_case_arrays(source,'case/ver','us')

    def test_warmup_cosine_optimizer_update_rates_and_restore(self):
        c={'max_updates_per_arm':1000,'learning_rate':1e-4,'minimum_learning_rate':1e-6,'lr_schedule':{'warmup_updates':50}}
        p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.Adam([p],lr=c['learning_rate'])
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda completed:learning_rate_multiplier(c,completed))
        rates=[];saved=None
        for step in range(1000):
            rates.append(opt.param_groups[0]['lr']);p.grad=torch.ones_like(p)
            opt.step();sch.step()
            if step==499:saved=copy.deepcopy((opt.state_dict(),sch.state_dict()))
        self.assertAlmostEqual(rates[0],2e-6,places=14)
        self.assertAlmostEqual(rates[49],1e-4,places=14)
        self.assertAlmostEqual(rates[-1],1e-6,places=14)
        self.assertTrue(all(a>=b for a,b in zip(rates[49:],rates[50:])))
        opt.load_state_dict(saved[0]);sch.load_state_dict(saved[1])
        self.assertAlmostEqual(opt.param_groups[0]['lr'],rates[500],places=14)
        self.assertEqual(sch.last_epoch,500)
        self.assertEqual(int(opt.state[p]['step']),500)

    def test_onecycle_actual_5000_updates_and_exact_resume(self):
        config={'max_updates_per_arm':5000,'lr_schedule':{'kind':'onecycle','pct_start':.1,'div_factor':25.,'final_div_factor':1000.}}
        for peak in (1e-3,):
            p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.Adam([p],lr=peak);sch=make_scheduler(opt,config)
            rates=[];saved=None
            for step in range(5000):
                rates.append(opt.param_groups[0]['lr']);p.grad=p.detach().clone()
                opt.step();sch.step()
                if step==2499:saved=copy.deepcopy((p.detach(),opt.state_dict(),sch.state_dict()))
            final=p.detach().clone()
            self.assertAlmostEqual(rates[0],peak/25,places=14)
            self.assertAlmostEqual(rates[499],peak,places=14)
            self.assertAlmostEqual(rates[-1],peak/25000,places=14)
            self.assertTrue(all(a<=b for a,b in zip(rates[:499],rates[1:500])))
            self.assertTrue(all(a>=b for a,b in zip(rates[499:],rates[500:])))
            self.assertEqual(opt.param_groups[0]['betas'],(.9,.999))
            with torch.no_grad():p.copy_(saved[0])
            opt.load_state_dict(saved[1]);sch.load_state_dict(saved[2])
            for step in range(2500,5000):
                self.assertEqual(opt.param_groups[0]['lr'],rates[step]);p.grad=p.detach().clone();opt.step();sch.step()
            self.assertTrue(torch.equal(p.detach(),final))
            self.assertEqual(int(opt.state[p]['step']),5000)

    def test_early_stopping_patience_relative_improvement_and_restore(self):
        settings={'min_updates':1500,'patience_updates':600,'min_relative_improvement':.001};state={}
        self.assertFalse(update_early_stopping(state,settings,0,10.))
        self.assertFalse(update_early_stopping(state,settings,900,9.))
        self.assertFalse(update_early_stopping(state,settings,1450,8.995))
        saved=copy.deepcopy(state)
        self.assertTrue(update_early_stopping(state,settings,1500,8.995))
        self.assertEqual(state['last_improvement_step'],900)
        self.assertFalse(update_early_stopping(saved,settings,1500,8.98))
        self.assertFalse(update_early_stopping(saved,settings,2050,8.98))
        self.assertTrue(update_early_stopping(saved,settings,2100,8.98))
        with self.assertRaises(FloatingPointError):update_early_stopping({},settings,1500,float('nan'))


if __name__=='__main__':unittest.main()
