"""Original canonical signed-U reference comparison after six-field rebarrier."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,csv,json,os,signal
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_U_reserve_accuracy_contract_v112 import approved_contract
    p,_=approved_contract(path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICAL_U_RESERVE_V112_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',phase='canonical_prediction_barrier',active=True,closed=False,execution=execution,owned_prediction_reads=0,alignment_array_reads=0,
        geometry_npz_reads=0,CSV_target_hash_reads=0,CSV_header_reads=0,CSV_target_parse_attempts=0,CSV_target_parses=0,original_U_case_reads=0,
        original_S_values_selected=0,raw_H5_deck_reads=0,K_PCG_replays=0,neural_PT_reads=0,neural_inference=0,training_updates=0,scaler_fits=0,FINAL_reads=0,
        GPU_execution=0,prediction_barrier_complete=False,fixed_field_denominator=6,completed_fields=0,metrics={},alignment_records={},prediction_inventory={},
        original_U_accuracy_gate_passed=False,new_OptiStruct_load_offnode_GT_accuracy=False,signed_S_solved=False,final_solver_selected=False)
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original300s earlier absolute deadline')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External originalU readout deadline')))
    try:
        import numpy as np
        def read_array(item,kind):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
            assert d.identity(item['path'],hashed=True)==expected
            v=np.load(d.path(item['path']),allow_pickle=False);a[kind]+=1
            dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
            assert list(v.shape)==item['shape'] and str(v.dtype)==dtype and np.isfinite(v).all();guard();return v
        fields={};predictions={}
        for cad in s['selection']['CADs']:
            row=s['cases'][cad];n=row['expected_geometry']['nodes']
            for load in s['selection']['loads']:
                item=row['physical_U_manifests'][load];assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path'])
                assert rec['key']==load and rec['initial_guess']=='independent_zero' and rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
                assert rec['all_original_nodes']==n;v=read_array(rec['physical_U'],'owned_prediction_reads');assert v.shape==(n,3) and v.dtype==np.float64
                key=cad+'/'+load;fields[key]=v;predictions[key]=dict(manifest_identity=item,physical_U_identity=rec['physical_U'],true_relative_residual=rec['true_relative_residual']);persist()
        assert set(fields)=={cad+'/'+load for cad in s['selection']['CADs']for load in s['selection']['loads']} and a['owned_prediction_reads']==6
        assert a['CSV_target_hash_reads']==a['CSV_header_reads']==a['CSV_target_parse_attempts']==0
        d.write(o/'prediction_barrier.json',dict(status='completed',all6_existing_predictions_finite_hash_body_Cartesian=True,source_GT_reads_before_barrier=0,inventory=predictions))
        a.update(prediction_barrier_complete=True,prediction_inventory=predictions,phase='original_CSV_U_only_readout');persist()
        for cad in s['selection']['CADs']:
            a['current_CAD']=cad;persist();row=s['cases'][cad];n=row['expected_geometry']['nodes']
            ids=read_array(row['GRID_identity'],'alignment_array_reads');permutation=read_array(row['H5_row_to_CSV_identity'],'alignment_array_reads')
            assert ids.dtype==permutation.dtype==np.int64 and ids.shape==permutation.shape==(n,)
            assert len(np.unique(ids))==n and np.array_equal(np.sort(permutation),np.arange(n))
            item=row['geometry_npz_identity'];assert d.identity(item['path'],hashed=True)==item
            with np.load(d.path(item['path']),allow_pickle=False)as z:xyz=z['vertices_mm']
            a['geometry_npz_reads']+=1;assert xyz.shape==(n,3) and xyz.dtype==np.float64 and np.isfinite(xyz).all()
            # The CSV is opened only after the complete six-prediction barrier.
            # Select GRID, physical XYZ and nine original signed U values. No
            # stress columns, encoded targets, scale fits or GT arrays output.
            guard();source=row['original_CSV_identity'];assert d.identity(source['path'],hashed=True)==source;a['CSV_target_hash_reads']+=1
            with d.path(source['path']).open('r',encoding='utf-8-sig',newline='')as f:
                header=next(csv.reader(f));a['CSV_header_reads']+=1
                names=s['reference']['columns'];assert all(header.count(name)==1 for name in names)
                a['CSV_target_parse_attempts']+=1;persist()
                values=np.loadtxt(f,delimiter=',',usecols=[header.index(name)for name in names],dtype=np.float64,ndmin=2)
                a['CSV_target_parses']+=1
            guard();assert values.shape==(n,13) and np.isfinite(values).all()
            aligned=values[permutation];assert np.array_equal(aligned[:,0],ids)
            distances=np.linalg.norm(aligned[:,1:4]-xyz,axis=1);maximum=float(distances.max())
            assert maximum<=s['reference']['coordinate_absolute_tolerance_mm']
            a['alignment_records'][cad]=dict(actual_allN=n,one_to_one=True,GRID_exact=True,maximum_CSV_geometry_distance_mm=maximum,
                original_CSV_identity=source,existing_mapping_preserved=True,target_CSV_parses=1,selected_signed_U_values=9*n,stress_values_selected=0)
            for j,load in enumerate(s['selection']['loads']):
                key=cad+'/'+load;reference=aligned[:,4+3*j:7+3*j];prediction=fields[key]
                difference=prediction-reference;norm=float(np.linalg.norm(reference));error=float(np.linalg.norm(difference));relative=error/norm if norm>0 else None
                rms=float(np.sqrt(np.square(difference).sum()/n));maximum=float(np.linalg.norm(difference,axis=1).max())
                passed=relative<=.01 if relative is not None else False
                a['metrics'][key]=dict(CAD=cad,load=load,all_original_nodes=n,U_relative_L2=relative,U_RMS_mm=rms,U_max_mm=maximum,
                    reference_norm=norm,zero_reference_norm=norm==0,true_relative_residual=predictions[key]['true_relative_residual'],passed=passed)
                a['original_U_case_reads']+=1;a['completed_fields']+=1;persist()
            del values,aligned,xyz,ids,permutation,reference,prediction,difference,distances
        assert a['CSV_target_hash_reads']==a['CSV_header_reads']==a['CSV_target_parse_attempts']==a['CSV_target_parses']==2
        assert a['geometry_npz_reads']==2 and a['alignment_array_reads']==4 and a['original_U_case_reads']==a['completed_fields']==6
        allnorm=all(v['U_relative_L2']is not None for v in a['metrics'].values())
        a.update(status='completed',phase='terminal',original_U_accuracy_gate_passed=all(v['passed']for v in a['metrics'].values()),
            complete6_mean_U_relative_L2=sum(v['U_relative_L2']for v in a['metrics'].values())/6 if allnorm else None,
            complete6_max_U_relative_L2=max(v['U_relative_L2']for v in a['metrics'].values())if allnorm else None,
            scientific_outcome='original_reserve_U_accuracy_PASS'if all(v['passed']for v in a['metrics'].values())else'fixed_original_reserve_U_accuracy_negative')
    except Exception as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
