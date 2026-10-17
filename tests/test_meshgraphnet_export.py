import copy,json,tempfile,unittest
from pathlib import Path
from tools.export_meshgraphnet_single_five import assemble,validate_scope_stop,merge_recovery

class MeshGraphExportTests(unittest.TestCase):
 def test_recovery_requires_complete_continuation_evidence(self):
  old={'arms':{'single_meshgraphnet':{'updates':5000},'five_meshgraphnet':{'updates':4}}}
  arm=dict(updates=5000,recovered_from_actual_step=4,added_updates=4996,model_Adam_scheduler_fourRNG_cursor_restored=True,resumed_full_node_backward_finite=True)
  recovered=dict(active=False,closed=True,status='completed',arms={'five_meshgraphnet':arm},source_reads=5,graph_reads=5,checkpoint_reads=1,stress_label_reads=0,heldout_reads=0,restored_without_readout_or_RNG_draw=True)
  terminal=dict(active=False,closed=True,exit_code=0,within_budget=True,original_deadline_respected=True)
  merged=merge_recovery(old,recovered,terminal);self.assertEqual(merged['arms']['five_meshgraphnet']['updates'],5000);self.assertEqual(old['arms']['five_meshgraphnet']['updates'],4)
  bad=copy.deepcopy(recovered);bad['arms']['five_meshgraphnet']['added_updates']=4995
  with self.assertRaisesRegex(ValueError,'step4'):merge_recovery(old,bad,terminal)
  bad=copy.deepcopy(terminal);bad['exit_code']=1
  with self.assertRaisesRegex(ValueError,'Successful closed'):merge_recovery(old,recovered,bad)
 def test_refuse_partial_and_unexpected_native_failure(self):
  c=json.loads(Path('configs/meshgraphnet_u_overfitting.json').read_text());s={'arms':{}}
  for g in ('single','five'):
   keys=c[g];s['arms'][g+'_meshgraphnet']=dict(status='completed',updates=5000,stop_reason='fixed_update_endpoint',actual_full_node_backward=True,output_channels=3,case_updates={k:5000//len(keys) for k in keys},endpoint_predictions_saved=len(keys),parameters=4603863,active_parameters=4603863,maximum_learning_rate=.001,learning_rate_last_update=4e-8,elapsed_seconds={'UTC':3.,'monotonic':3.},model_config={'message_passing_steps':15,'latent_size':180,'hidden_layers':2,'teacher_or_previous_displacement':False,'sample_id':False,'node_sampling':False},endpoint={'step':5000,'per_case':{k:{'U_percent':2.} for k in keys},'mean_U_percent':2.,'worst_field_percent':2.})
  r,p=assemble(s,c);self.assertEqual((len(r),len(p)),(2,6))
  bad=copy.deepcopy(s);bad['arms']['five_meshgraphnet']['updates']=4999
  with self.assertRaisesRegex(ValueError,'Complete actual5000'):assemble(bad,c)
  s.update(status='failed',active=False,closed=True,error_type='FileExistsError',error='fifty_meshgraphnet already exists')
  terminal=dict(active=False,closed=True,timeout=False,original_deadline_respected=True,elapsed_seconds={'UTC':5.,'monotonic':5.},exit_code=1,within_budget=False)
  scope={'selected_arms':['single_meshgraphnet','five_meshgraphnet']}
  with tempfile.TemporaryDirectory() as folder:
   run=Path(folder);(run/'fifty_meshgraphnet').mkdir();(run/'fifty_meshgraphnet/USER_SCOPE_GATE.json').write_text('{}')
   self.assertTrue(validate_scope_stop(s,terminal,scope,run,c));self.assertEqual(terminal['exit_code'],1);self.assertFalse(terminal['within_budget'])
   bad=copy.deepcopy(s);bad['error_type']='MemoryError'
   with self.assertRaisesRegex(ValueError,'technical failures'):validate_scope_stop(bad,terminal,scope,run,c)
   bad=copy.deepcopy(s);bad['arms']['fifty_meshgraphnet']={'updates':0}
   with self.assertRaisesRegex(ValueError,'Only requested'):validate_scope_stop(bad,terminal,scope,run,c)

if __name__=='__main__':unittest.main()
