import contextlib,sys,types,unittest
from unittest.mock import patch
import torch
from geobc.overfit_tcnn import TCNNField,TCNNFieldConfig

class PrecisionTests(unittest.TestCase):
 def test_backend_config(self):
  self.assertEqual(TCNNFieldConfig().network_config()['otype'],'FullyFusedMLP')
  self.assertEqual(TCNNFieldConfig(native_precision='fp32').network_config()['otype'],'CutlassMLP')
  with self.assertRaises(ValueError):TCNNFieldConfig(native_precision='tf32')
 def test_native_half_cannot_claim_fp32(self):
  fake=types.SimpleNamespace(NetworkWithInputEncoding=lambda **kw:types.SimpleNamespace(dtype=torch.float16))
  with patch.dict(sys.modules,{'tinycudann':fake}),patch('torch.cuda.device',return_value=contextlib.nullcontext()):
   with self.assertRaisesRegex(RuntimeError,'native dtype'):
    TCNNField(config=TCNNFieldConfig(native_precision='fp32'))

if __name__=='__main__':unittest.main()
