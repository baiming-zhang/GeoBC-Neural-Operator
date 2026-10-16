import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.prepare_fifty_u_overfitting import selection,ALIGNMENT
from tools.train_sample_overfitting import cache_case_keys,read_training_source
from tools.export_fifty_u_overfitting import assemble,NAMES


class FiftyTests(unittest.TestCase):
    def test_selection_nested_unique_and_excludes_nontrain(self):
        anchors=[f'a{i}/ver' for i in range(5)]
        pool=[f'b{i}/hor' for i in range(50)]
        keys=selection(anchors,pool,[f'b{i}' for i in range(37)],[f'c{i:02}' for i in range(60)])
        self.assertEqual(keys[:5],anchors);self.assertEqual(len({k.split('/')[0] for k in keys}),50)
        self.assertFalse(any(k.startswith('b37/') for k in keys))
        self.assertEqual(cache_case_keys(dict(groups=['fifty'],fifty=keys)),keys)
        self.assertEqual(cache_case_keys(dict(single=anchors[:1],five=anchors)),anchors)

    def test_interleaved_train_U_source_ignores_stress_column(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            for key,array in dict(xyz=np.zeros((3,3)),features=np.ones((3,25)),physical=np.column_stack((np.arange(9).reshape(3,3),np.full(3,np.nan)))).items():np.save(d/(key+'.npy'),array)
            source=dict(sample_key='case/ver',scientific_group='TRAIN',label_alignment=ALIGNMENT,**{k+'_path':str(d/(k+'.npy')) for k in ('xyz','features','physical')})
            xyz,features,u=read_training_source(source,'case/ver','u')
            np.testing.assert_array_equal(u,np.arange(9).reshape(3,3));self.assertTrue(np.isfinite(u).all())
            for bad in (dict(source,scientific_group='FINAL'),dict(source,sample_key='other/ver')):
                with self.assertRaises(ValueError):read_training_source(bad,'case/ver','u')
            with self.assertRaises(ValueError):read_training_source(source,'case/ver','us')

    def test_export_rejects_incomplete_visits_and_unequal_recipe(self):
        config=json.loads(Path('configs/fifty_u_overfitting.json').read_text());previous=json.loads(Path('results/fullcycle_u_overfitting/metrics.json').read_text())
        summary={'arms':{}}
        for method in NAMES:
            keys=config['fifty'];m=dict(step=5000,mean_U_percent=2.,worst_field_percent=2.,per_case={k:dict(U_percent=2.) for k in keys})
            summary['arms']['fifty_'+method]=dict(status='completed',updates=5000,endpoint=m,stop_reason='fixed_update_endpoint',actual_full_node_backward=True,output_channels=3,endpoint_predictions_saved=50,case_updates={k:100 for k in keys},parameters=4600000,active_parameters=4600000,maximum_learning_rate=.001,learning_rate_last_update=4e-8,elapsed_seconds=dict(UTC=10.,monotonic=9.))
        rows,cases=assemble(summary,config,previous);self.assertEqual((len(rows),len(cases)),(8,400))
        bad=copy.deepcopy(summary);bad['arms']['fifty_mlp']['case_updates'][keys[0]]=99
        with self.assertRaisesRegex(ValueError,'100 complete'):assemble(bad,config,previous)
        bad=copy.deepcopy(summary);bad['arms']['fifty_mlp']['updates']=4999
        with self.assertRaisesRegex(ValueError,'actual5000'):assemble(bad,config,previous)
        bad=copy.deepcopy(config);bad['learning_rate']=1e-4
        with self.assertRaisesRegex(ValueError,'Same common'):assemble(summary,bad,previous)


if __name__=='__main__':unittest.main()
