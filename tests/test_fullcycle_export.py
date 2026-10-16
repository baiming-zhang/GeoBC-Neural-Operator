import copy,json,unittest
from pathlib import Path
from tools.export_fullcycle_u_overfitting import assemble,NAMES


class FullCycleExportTests(unittest.TestCase):
    def test_replace_all_early_stops_retain_complete_endpoints(self):
        original=json.loads(Path('configs/uniform_onecycle_u_overfitting.json').read_text())
        config=json.loads(Path('configs/fullcycle_u_overfitting.json').read_text())
        selected=['single_hashgrid','single_tcnn_hash','five_mlp','five_ngp4'];config['selected_arms']=selected
        base={'arms':{}};rerun={'arms':{}}
        for group in ('single','five'):
            for method in NAMES:
                name=group+'_'+method
                endpoint=dict(step=5000,per_case={k:dict(U_percent=2.) for k in config[group]},mean_U_percent=2.,worst_field_percent=2.)
                arm=dict(status='completed',method=method,group=group,updates=5000,endpoint=endpoint,output_channels=3,actual_full_node_backward=True,parameters=4600000,active_parameters=4600000,endpoint_predictions_saved=len(config[group]),maximum_learning_rate=1e-3,learning_rate_last_update=4e-8,elapsed_seconds=dict(UTC=10.,monotonic=9.),stop_reason='update_cap')
                base['arms'][name]=arm
                if name in selected:
                    rerun['arms'][name]=copy.deepcopy(arm)
                    arm.update(updates=1500,stop_reason='early_stopping_plateau');arm['endpoint']['step']=1500
                    arm['endpoint'].update(mean_U_percent=99.,worst_field_percent=99.)
        rows,cases=assemble(base,rerun,original,config)
        self.assertEqual(len(rows),16);self.assertEqual(len(cases),48)
        self.assertTrue(all(r['updates']==5000 and r['U_percent']==2. for r in rows))
        self.assertEqual(sum(r['fresh_rerun_of_early_stop'] for r in rows),4)
        broken=copy.deepcopy(rerun);broken['arms']['five_mlp']['updates']=4999
        with self.assertRaisesRegex(ValueError,'actual5000'):assemble(base,broken,original,config)
        missing=copy.deepcopy(rerun);del missing['arms']['five_mlp']
        with self.assertRaisesRegex(ValueError,'exactly once'):assemble(base,missing,original,config)
        unequal=copy.deepcopy(config);unequal['learning_rate']=1e-4
        with self.assertRaisesRegex(ValueError,'Same common recipe'):assemble(base,rerun,original,unequal)


if __name__=='__main__':unittest.main()
