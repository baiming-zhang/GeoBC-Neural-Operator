"""Scalar-only fixture: reject incomplete runs and never substitute best checkpoints."""
import csv,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.export_onecycle_u_overfitting import run


class OneCycleExportTests(unittest.TestCase):
    def test_variable_stop_endpoints_all16_and_reject_time_cap(self):
        config=json.loads(Path('configs/uniform_onecycle_u_overfitting.json').read_text())
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            previous=Path.cwd();base=Path(directory);root=base/'run';root.mkdir()
            arms={}
            for group in ('single','five'):
                for method in config['methods']:
                    name=group+'_'+method;folder=root/name;folder.mkdir()
                    early=method=='transolver';step=1500 if early else 5000
                    endpoint=dict(step=step,per_case={k:dict(U_percent=2.) for k in config[group]},mean_U_percent=2.,worst_field_percent=2.)
                    arms[name]=dict(status='completed',method=method,group=group,output_channels=3,updates=step,endpoint=endpoint,actual_full_node_backward=True,parameters=4600000,active_parameters=4600000,endpoint_predictions_saved=len(config[group]),maximum_learning_rate=config.get('learning_rate_by_method',{}).get(method,config['learning_rate']),learning_rate_last_update=1e-6,elapsed_seconds=dict(UTC=10.,monotonic=9.),best_step=50,best=dict(worst_field_percent=.1),stop_reason='early_stopping_plateau' if early else 'update_cap',early_stopping=dict(checked_step=step,updates_without_improvement=600))
                    (folder/'metrics.jsonl').write_text(json.dumps(endpoint)+'\n')
            summary=dict(status='completed',active=False,closed=True,source_reads=5,stress_label_reads=0,heldout_reads=0,arms=arms)
            (root/'summary.json').write_text(json.dumps(summary))
            (root/'terminal.json').write_text(json.dumps(dict(exit_code=0,within_budget=True,active=False,closed=True,elapsed_seconds=dict(UTC=100.,monotonic=90.))))
            (root/'launch.json').write_text(json.dumps(dict(entry_epoch=1000.)))
            config_path=base/'config.json';config_path.write_text(json.dumps(config))
            os.chdir(base)
            try:
                with patch('tools.export_onecycle_u_overfitting.write_plot'):
                    run(root,config_path)
                    with (base/config['report_directory']/'summary.csv').open() as f:rows=list(csv.DictReader(f))
                    with (base/config['report_directory']/'per_case.csv').open() as f:cases=list(csv.DictReader(f))
                    self.assertEqual(len(rows),16);self.assertEqual(len(cases),48)
                    self.assertTrue(all(float(r['U_percent'])==2. for r in rows))
                    self.assertEqual(next(r for r in rows if r['experiment']=='single_transolver')['updates'],'1500')
                    config['learning_rate_by_method']={'transolver':1e-4};config_path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError,'Uniform5000'):run(root,config_path)
                    config.pop('learning_rate_by_method');config_path.write_text(json.dumps(config))
                    summary['arms']['single_transolver']['stop_reason']='fixed_arm_time_cap'
                    (root/'summary.json').write_text(json.dumps(summary))
                    with self.assertRaisesRegex(ValueError,'Unexpected stop'):run(root,config_path)
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
