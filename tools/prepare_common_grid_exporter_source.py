"""Freeze CPU-only reporter after full source review, without running it."""
import ast
from geobc import available_cad_pilot_data as d


def run():
    own='tools/export_common_grid_comparison.py';ast.parse(d.path(own).read_text(encoding='utf-8-sig'))
    plan=d.read('notes/geobc_common_grid_readout_plan.json')
    baseline=d.read('results/geobc_available_cad_pilot/metrics.json')
    assert baseline['scientific_qualification'] and baseline['case_keys']==plan['scientific']['case_keys']
    for key in ('fno_cad40_fresh4800','transolver_pp_cad40_fresh4800'):
        row=baseline['rows'][key]
        assert all(name in row for name in ('full60','inherited_training_seconds','actual_inference_seconds','total_parameters'))
        assert row['full60']['ranking_eligible'] and row['full60']['primary_score'] is not None
    marker={'passed':True,'reviewer':'root','whole_source_reviewed':True,
        'actual_export_released':True,'conditional_on_genuine_training_and_readout_terminal':True,
        'exporter_source_sha256':d.sha(own),'scientific_digest':plan['scientific_digest'],
        'actual_export_executions':0,'actual_arrays_GT_PT_GPU':0,
        'review_basis':['Reporter only imports stdlib/deferred CPU Matplotlib, saved JSON/source checks/exact /proc exit checks',
                       'Genuine native/external terminal and budgetqualification; original120 allfinite barrier and60 cachedGT count required',
                       'Negative cancels new leader/DEV metrics, full120/60 denominator retained; old rows explicitly historical references',
                       'TRAIN120 versus knownDEV60 separate, epoch/monotonic costs separate, missing costs null/no zero fill',
                       'Both new rows brief; full4 row CSV/JSON and perload/paired CAD details; ties kept',
                       'Exports once to fresh destination and required PNG; source-ready is not result completion'],
        'final_solver_selected':False}
    d.write('notes/geobc_common_grid_exporter_root_review.json',marker)
    print({'status':'CPU_reporter_source_ready','exporter_sha256':marker['exporter_source_sha256'],'actual_export':0})


if __name__=='__main__':run()
