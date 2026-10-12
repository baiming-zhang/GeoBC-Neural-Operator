"""One CPU scalar-only report after real complete fixed100 source terminal."""
import argparse,time,csv,io
from geobc import available_cad_pilot_data as d
from geobc.transolver_decoder_contract_v64 import terminal
PLAN='notes/geobc_main_train_source_qualification_v122_plan.json'
REVIEW='notes/geobc_main_train_source_qualification_v122_exporter_root_review.json'
OUTPUT='results/geobc_main_train_source_qualification_v122'

def export():
    began=time.perf_counter();p,a,e,b=terminal(PLAN);r=d.read(REVIEW)
    assert r['passed'] and r['whole_source_reviewed'] and r['conditional_actual_export_released'] and r['source_sha256']==d.sha('tools/export_geobc_main_train_sources_v122.py')
    assert r['SCI']==p['scientific_digest'] and a['status']=='completed' and a['source_barrier_complete'] and a['source_qualification_passed']
    assert a['committed_CADs']==100 and a['raw_U_S_target_reads']==a['GT_reads']==a['DEV_FINAL_reads']==a['K_PCG_GPU_reads']==0
    receipt=d.read('notes/geobc_main_train_source_qualification_v122_actual_terminal_receipt.json')
    assert receipt['actual_exec_exit_code']==0 and receipt['exact_processes_exited'] and receipt['terminal_binding']==b and e['within_external_budget']
    o=d.path(OUTPUT);assert not o.exists();o.mkdir()
    rows=[]
    for cad in p['scientific']['selection']['CADs']:
        c=a['case_records'][cad];g=c['expected_geometry'];rows.append(dict(CAD=cad,original_nodes=g['nodes'],original_cells=g['cells'],source_epoch_seconds=c['actual_source_wall_epoch'],source_monotonic_seconds=c['actual_source_wall_monotonic'],reused_accepted_source_geometry=c['inherited_geometry'],canonical_loads=3,actual_K_PCG_or_feature_qualification=False,GT_reads=0))
    assert len(rows)==100 and len({x['CAD'] for x in rows})==100
    with (o/'sources.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    whole=dict(UTC=e['elapsed_epoch_seconds'],monotonic=e['elapsed_monotonic_seconds']);counts={k:a[k] for k in ('deck_hash_reads','H5_hash_reads','H5_input_reads','GRID_archive_member_reads','permutation_reads','CSV_GRID_XYZ_reads','legacy_cache_reads','deck_parses','deck_header_reads')}
    d.write(o/'metrics.json',dict(status='completed',SCI=p['scientific_digest'],source_gate_passed=True,fixed100_CADs=100,canonical_RHS_count=300,actual_counts=counts,whole_external_seconds=whole,actual_K_PCG_or_feature_qualification=False,neural_training_released=False,GT_reads=0,rows=rows,
        limitations=['Source/schema/load transfer qualification only; no measured physicalU/gradient input readiness or S accuracy','TRAIN-only cohort/shape-family unproven/approxP2 source controls not exactOptiStruct']))
    (o/'BRIEF.md').write_text(f'# Main TRAIN100 target-free source qualification\n\nFixed100 CADs/300 canonical loads qualified. Whole UTC {whole["UTC"]:.3f}s / monotonic {whole["monotonic"]:.3f}s; clocks retained separately.\n\n96 new H5 vertices/cells and 4 accepted source caches;36 legacy GRID-only members and60 original CSV GRID/XYZ-only reads. Canonical real LOAD expansion/material/zero supports checked; source caches are local. No target, K/PCG/GPU/model or DEV/FINAL reads.\n\nThis is source compatibility, not physical U/gradient production, stress accuracy, heldout generalization or exact OptiStruct fidelity. Next: independently frozen GPU U/full-gradient input production with accepted legacy endpoints/operator reuse.\n',encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    x=range(100);axes[0].plot(x,[v['source_epoch_seconds'] for v in rows],label='UTC phase cost');axes[0].plot(x,[v['source_monotonic_seconds'] for v in rows],label='Monotonic phase cost',alpha=.75)
    axes[0].set(xlabel='Fixed TRAIN CAD index',ylabel='Seconds',title='Source/cache preparation');axes[0].legend(fontsize=8)
    axes[1].scatter([v['original_nodes']/1000 for v in rows],[v['original_cells']/1000 for v in rows],s=15);axes[1].set(xlabel='Original nodes (thousands)',ylabel='Original TET10 cells (thousands)',title='Complete100 geometry schemas')
    figure.suptitle('Target-free sources:100/100; canonical RHS:300/300\nK / PCG / kinematic features / GT: not executed',fontsize=12)
    figure.savefig(o/'comparison.png',dpi=160);plt.close(figure)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,CSV_rows=100,source_gate_passed=True,PNG_visual_review_pending=True,actual_new_GT_PT_arrays_GPU=0,files=['BRIEF.md','sources.csv','metrics.json','comparison.png']))
    print(dict(status='actual_CPU_export_completed',CSV_rows=100,actual_seconds=time.perf_counter()-began))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"SOURCE_only","actual_exports_GT_arrays_GPU":0}')
