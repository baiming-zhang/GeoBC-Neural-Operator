"""Reject mislabeled precision and partial endpoints before publishing scalars."""
import copy,json,unittest
from pathlib import Path
from tools.export_tcnn_fp32_u_overfitting import assemble

class ExportPrecisionTests(unittest.TestCase):
 def test_reject_partial_and_native_half(self):
  c=json.loads(Path('configs/tcnn_fp32_u_overfitting.json').read_text())
  s={'precision_runtime':{'allow_tf32':False,'NVIDIA_TF32_OVERRIDE':'0'},'arms':{}}
  for g in ('single','five','fifty'):
   keys=c[g];s['arms'][g+'_tcnn_hash']=dict(status='completed',updates=5000,stop_reason='fixed_update_endpoint',actual_full_node_backward=True,case_updates={k:5000//len(keys) for k in keys},endpoint_predictions_saved=len(keys),model_config={'actual_native_dtype':'torch.float32','actual_native_hyperparams':{'network':{'otype':'CutlassMLP'}}},parameters=4597120,active_parameters=4597120,maximum_learning_rate=.001,learning_rate_last_update=4e-8,elapsed_seconds={'UTC':3.,'monotonic':3.},endpoint={'step':5000,'per_case':{k:{'U_percent':2.} for k in keys},'mean_U_percent':2.,'worst_field_percent':2.})
  rows,cases=assemble(s,c);self.assertEqual((len(rows),len(cases)),(3,56))
  bad=copy.deepcopy(s);bad['arms']['fifty_tcnn_hash']['updates']=4999
  with self.assertRaisesRegex(ValueError,'Complete5000'):assemble(bad,c)
  bad=copy.deepcopy(s);bad['arms']['single_tcnn_hash']['model_config']['actual_native_dtype']='torch.float16'
  with self.assertRaisesRegex(ValueError,'Actual native FP32'):assemble(bad,c)
  bad=copy.deepcopy(s);bad['precision_runtime']['allow_tf32']=True
  with self.assertRaisesRegex(ValueError,'Runtime precision'):assemble(bad,c)

if __name__=='__main__':unittest.main()
