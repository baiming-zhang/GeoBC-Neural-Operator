"""One-shot CPU target preparation for the frozen additional TRAIN60 only."""
import argparse
import csv
import json
import os
import time
from geobc import available_cad_pilot_data as d

PLAN = 'notes/geobc_prospective_train_targets_plan.json'


def approved():
    plan = d.read(PLAN)
    review = d.read(plan['root_review'])
    assert review['passed'] and review['CPU_preparation_released']
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert review['source_pins'] == plan['source_pins']
    assert plan['scientific']['whole_CPU_budget_seconds'] == 1800
    assert all(d.sha(name) == sha for name, sha in plan['source_pins'].items())
    for name in ('split_identity', 'input_binding_identity'):
        item = plan['scientific'][name]
        assert d.identity(item['path'], True) == item
    split = d.read(plan['scientific']['split_identity']['path'])
    binding = d.read(plan['scientific']['input_binding_identity']['path'])
    assert binding['scientific']['split_scientific_digest'] == split['scientific_digest']
    selected = plan['scientific']['selected_TRAIN60_CADs']
    assert selected == split['scientific']['additional_TRAIN60_CADs'] and len(set(selected)) == 60
    excluded = set(split['scientific']['DEV_CADs'] + split['scientific']['FINAL_CADs'] +
                   split['scientific']['old_TRAIN40_CADs'] + split['scientific']['original_sealed_DEV20_preserved'])
    assert not set(selected) & excluded
    assert plan['scientific']['scaler'] == split['scientific']['scaler']
    helper = binding['scientific']['helper_contract']
    sources = {cad: {component: helper['raw_sources'][cad][component]
                     for component in ('Field', 'FieldMesh')} for cad in selected}
    protected_paths = {source['path'] for cad in split['scientific']['FINAL_CADs']
                       for source in helper['raw_sources'][cad].values()}
    assert all(source['path'] not in protected_paths for row in sources.values() for source in row.values())
    assert all(source['prospective_role'] == 'TRAIN_input_only' for row in sources.values() for source in row.values())
    return plan, sources


def run():
    plan, sources = approved()
    output = d.path(plan['output'])
    assert not any((output/name).exists() for name in ('execution.json','summary.json','clock.json','cases','alignment'))
    stop = time.monotonic()+2
    while not (output/'external_launch.json').exists():
        if time.monotonic() >= stop: raise RuntimeError('External launch identity missing')
        time.sleep(.01)
    external = d.read(output/'external_launch.json')
    own = d.process_snapshot(os.getpid())
    assert external['scientific_digest'] == plan['scientific_digest']
    assert own and all(own[key] == external['child_process'][key] for key in ('pid','start_ticks','argv'))
    started_epoch, started_mono = external['started_epoch'], external['started_monotonic']
    result = {'status':'running','scientific_digest':plan['scientific_digest'], 'run_process':own,
              'selected_TRAIN60_CADs':plan['scientific']['selected_TRAIN60_CADs'],
              'fixed_CAD_denominator':60,'fixed_case_denominator':180,
              'CPU_only':True,'records':{},'CADs':{},'scaler':plan['scientific']['scaler'],
              'read_receipt':{'H5_vertices_reads':0,'CSV_coordinate_and_known3_parse_attempts':0,
                              'CSV_coordinate_and_known3_parses':0,'TRAIN_U_case_reads':0,'TRAIN_S_case_reads':0,
                              'DEV_target_parses':0,'FINAL_target_parses':0,'H5_nodal_variable_reads':0,
                              'source_strain_reads':0,'dia_target_reads':0,'scaler_fits':0,'raw_byte_hashes':0},
              'new_input_feature_grid_preparation':0,'new_model_training':0,
              'GPU_work':0,'CUDA_initialized':False,'clock':{}}
    d.write(output/'execution.json', {'scientific_digest':plan['scientific_digest'],
                                    'run_process':own,'external_guard':external['guard_process']})

    def persist(phase, closed=False):
        result['clock'] = {'started_epoch':started_epoch,'started_monotonic':started_mono,
                           'elapsed_epoch_seconds':time.time()-started_epoch,
                           'elapsed_monotonic_seconds':time.monotonic()-started_mono,
                           'active':not closed,'closed':closed,'phase':phase}
        d.write(output/'clock.json', result['clock'])
        d.write(output/'status.json', {'status':result['status'],'phase':phase,
                                      'complete_CADs':len(result['CADs']), 'complete_cases':len(result['records']),
                                      'clock':result['clock']})
        d.write(output/'manifest.json', result)

    def guard():
        if min(started_epoch+1800-time.time(), started_mono+1800-time.monotonic()) <= 0:
            raise TimeoutError('Original whole CPU preparation budget exhausted')

    def source_hash(item):
        d.stat_matches(item)
        import hashlib
        value = hashlib.sha256()
        with d.path(item['path']).open('rb') as stream:
            for block in iter(lambda:stream.read(4*2**20),b''):
                guard(); value.update(block)
        result['read_receipt']['raw_byte_hashes'] += 1
        assert value.hexdigest() == item['sha256'], 'Source byte identity differs'
        d.stat_matches(item)

    def save(filename, value):
        filename.parent.mkdir(parents=True,exist_ok=True)
        temporary = filename.with_suffix(filename.suffix+'.tmp')
        with temporary.open('xb') as stream: np.save(stream,value,allow_pickle=False)
        temporary.replace(filename)
        return d.identity(filename,True)

    persist('imports')
    try:
        import numpy as np
        import h5py
        from scipy.spatial import cKDTree
        import torch
        from geobc.benchmark import FieldScaler
        torch.set_num_threads(1)
        result['CUDA_initialized'] = torch.cuda.is_initialized()
        if result['CUDA_initialized']: result['GPU_work'] = None
        assert not result['CUDA_initialized']
        scaler = FieldScaler(**plan['scientific']['scaler'])  # No fit call.
        result['library_versions'] = {'numpy':np.__version__,'h5py':h5py.__version__,'torch':torch.__version__}
        assert plan['scientific']['loads'] == d.LOADS
        for cad in plan['scientific']['selected_TRAIN60_CADs']:
            guard(); persist('source_identity_'+cad)
            for item in sources[cad].values(): source_hash(item)
            mesh = sources[cad]['FieldMesh']; field = sources[cad]['Field']
            persist('TRAIN_vertices_'+cad)
            with h5py.File(d.path(mesh['path']),'r') as handle:
                vertices = np.asarray(handle['vertices'][:],dtype=np.float32)
            result['read_receipt']['H5_vertices_reads'] += 1
            assert vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) and np.isfinite(vertices).all()
            d.stat_matches(mesh)
            with d.path(field['path']).open(encoding='utf-8-sig',newline='') as stream:
                columns = next(csv.reader(stream))
            names = d.COORDS+d.TARGET_COLUMNS
            assert len(columns) == len(set(columns)) and all(name in columns for name in names)
            result['read_receipt']['CSV_coordinate_and_known3_parse_attempts'] += 1
            persist('TRAIN_numeric_parse_'+cad)
            values = np.loadtxt(d.path(field['path']),delimiter=',',skiprows=1,
                                usecols=[columns.index(name) for name in names],dtype=np.float64,ndmin=2)
            result['read_receipt']['CSV_coordinate_and_known3_parses'] += 1
            result['read_receipt']['TRAIN_U_case_reads'] += 3
            result['read_receipt']['TRAIN_S_case_reads'] += 3
            persist('TRAIN_alignment_'+cad)
            assert values.shape == (len(vertices),16) and np.isfinite(values).all()
            ids = values[:,0]
            assert (ids > 0).all() and (ids < 2**53).all() and np.array_equal(ids,np.rint(ids))
            assert len(np.unique(ids)) == len(ids)
            distance, permutation = cKDTree(values[:,1:4]).query(vertices,workers=1)
            assert len(np.unique(permutation)) == len(vertices) and float(distance.max()) <= .0002
            permutation = np.asarray(permutation,np.int64)
            aligned = np.ascontiguousarray(values[permutation,4:],dtype=np.float32)
            assert aligned.shape == (len(vertices),12) and np.isfinite(aligned).all()
            permutation_identity = save(output/'alignment'/cad/'h5_row_to_csv.npy',permutation)
            cad_records = {}
            for index,load in enumerate(d.LOADS):
                guard()
                key = cad+'/'+load
                physical = np.ascontiguousarray(aligned[:,4*index:4*index+4])
                encoded = scaler.encode(physical)
                assert encoded.shape == physical.shape and np.isfinite(encoded).all()
                directory = output/'cases'/key.replace('/','_')
                physical_id = save(directory/'physical.npy',physical)
                target_id = save(directory/'target.npy',encoded)
                norms = physical.astype(np.float64)
                cad_records[key] = {'key':key,'cad_id':cad,'scientific_group':'TRAIN','status':'completed',
                    'num_nodes':len(vertices),'physical_path':physical_id['path'],'target_path':target_id['path'],
                    'file_identities':{'physical_path':physical_id,'target_path':target_id},
                    'permutation_identity':permutation_identity,'source':sources[cad],
                    'label_alignment':'official CSV GRID coordinates -> H5 vertices',
                    'physical_channels':['Ux(mm)','Uy(mm)','Uz(mm)','original signed S(MPa)'],
                    'displacement_squared_norm':float(np.square(norms[:,:3]).sum()),
                    'stress_squared_norm':float(np.square(norms[:,3]).sum())}
            d.stat_matches(field)
            result['records'].update(cad_records)
            result['CADs'][cad] = {'num_nodes':len(vertices),'CSV_H5_max_distance_mm':float(distance.max()),
                                   'permutation_identity':permutation_identity,'case_keys':list(cad_records)}
            assert not torch.cuda.is_initialized()
            persist('complete_TRAIN_'+cad)
        guard()
        assert set(result['records']) == set(d.keys(plan['scientific']['selected_TRAIN60_CADs']))
        assert result['read_receipt']['CSV_coordinate_and_known3_parses'] == 60
        assert result['read_receipt']['TRAIN_U_case_reads'] == result['read_receipt']['TRAIN_S_case_reads'] == 180
        assert result['read_receipt']['H5_vertices_reads'] == 60 and result['read_receipt']['raw_byte_hashes'] == 120
        assert all(result['read_receipt'][name] == 0 for name in ('DEV_target_parses','FINAL_target_parses',
            'H5_nodal_variable_reads','source_strain_reads','dia_target_reads','scaler_fits'))
        assert len(result['CADs']) == 60
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        if 'torch' in locals():
            result['CUDA_initialized'] = torch.cuda.is_initialized()
            if result['CUDA_initialized']: result['GPU_work'] = None
        persist('terminal',closed=True)
        d.write(output/'summary.json',result)
    print(json.dumps({'status':result['status'],'complete_CADs':len(result['CADs']),
                      'complete_cases':len(result['records']),'clock':result['clock'],
                      'read_receipt':result['read_receipt'],'new_GPU_or_training':0}))
    if result['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
