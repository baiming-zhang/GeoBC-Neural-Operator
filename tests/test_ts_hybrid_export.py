import copy,json,unittest
from pathlib import Path
from tools.export_transolver_hybrid_ablations import assemble

class HybridExportTests(unittest.TestCase):
 def test_reject_partial_and_missing_case(self):
  c=json.loads(Path('configs/transolver_hybrid_u_overfitting.json').read_text());s={'precision_runtime':{'allow_tf32':True,'NVIDIA_TF32_OVERRIDE':None},'arms':{}}
  for g in ('single','five','fifty'):
   keys=c[g];s['arms'][g+'_transolver_hybrid']=dict(status='completed',updates=5000,stop_reason='fixed_update_endpoint',actual_full_node_backward=True,output_channels=3,case_updates={k:5000//len(keys) for k in keys},endpoint_predictions_saved=len(keys),model_config={'query_head':[44,376,376,3],'latent_channels':16,'label_input':False,'backbone':{'method':'transolver'}},parameters=4574719,active_parameters=4574719,maximum_learning_rate=.001,learning_rate_last_update=4e-8,elapsed_seconds={'UTC':3.,'monotonic':3.},endpoint={'step':5000,'per_case':{k:{'U_percent':2.} for k in keys},'mean_U_percent':2.,'worst_field_percent':2.})
  r,p=assemble(s,c,'transolver_hybrid','fixture');self.assertEqual((len(r),len(p)),(3,56))
  bad=copy.deepcopy(s);bad['arms']['fifty_transolver_hybrid']['updates']=4999
  with self.assertRaisesRegex(ValueError,'Complete5000'):assemble(bad,c,'transolver_hybrid','fixture')
  bad=copy.deepcopy(s);bad['arms']['five_transolver_hybrid']['case_updates'][c['five'][0]]=999
  with self.assertRaisesRegex(ValueError,'full visits'):assemble(bad,c,'transolver_hybrid','fixture')

if __name__=='__main__':unittest.main()
