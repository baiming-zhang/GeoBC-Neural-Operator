"""Only metadata/source/file stats, no PT/arrays/raw numeric labels or GPU."""
import argparse
from pathlib import Path
from geobc import available_cad_pilot_data as d
from geobc.reserve_holdout_contract_v93 import PLAN,REVIEW,DESIGN,SOURCES,terminal


def prepare():
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    cads=['50_380','624_483'];split=d.read('notes/geobc_prospective_cad_split_v1.json')
    assert split['scientific']['reserve_CADs_not_substitutes']==cads
    old=d.read('data/geobc_prospective_dataset_v1/manifest.json')
    assert all(cad not in split['scientific'][role] for cad in cads for role in ('TRAIN_CADs','DEV_CADs','FINAL_CADs','excluded_tracked_history_CADs'))
    assert not any(k.split('/')[0]in cads for k in old['records'])
    # Recorded realized input/label caches, including legacy prepared files and
    # all project run arrays. Raw download presence and future TRAIN200 intent
    # do not count as realized numerical exposure.
    matches=[]
    for root in (d.path('data'),d.path('runs')):
        for file in root.rglob('*'):
            if not file.is_file() or file.suffix not in ('.npy','.npz'):continue
            text=d.relative(file)
            if any(cad in text for cad in cads):matches.append(text)
    assert not matches
    scope_manifests=['data/geobc_available_cad_pilot/manifest.json','data/geobc_prospective_dataset_v1/manifest.json']
    for file in scope_manifests:
        m=d.read(file);assert not any(k.split('/')[0]in cads for k in m['records'])
    evidence=dict(status='metadata_inventory_completed',observed_at_utc=d.utc(),CADs=cads,
        split_identity=d.identity('notes/geobc_prospective_cad_split_v1.json',hashed=True),
        realized_input_scope_identities=[d.identity(x,hashed=True)for x in scope_manifests],
        array_filename_matches=matches,unexposed_by_recorded_execution_metadata=True,
        history_search='Both IDs in earlier summaries occur only in planned TRAIN200/source-pool lists, not realized cases or read ledgers',
        limits=['Metadata evidence is not proof of an arbitrary external access never occurring','Both were already reserved without performance selection; acquisition availability conditional','Only2 CADs/6 loads, not a final population estimate'])
    evidence_path='notes/geobc_reserve_holdout_v93_exposure_evidence.json';d.write(evidence_path,evidence)
    ref=d.read('runs/geobc_exploration/FNO100_knownDEV60_physical30_40_v73/summary.json')['candidates']['FNO100_PHYSICAL_6000']
    candidates=[]
    for name,family,planfile,cp in [
        ('FNO20','FNO9','notes/geobc_fno_learning20_v56_plan.json',ref['checkpoint']),
        ('FNO_HASH20','FNO_GLOBAL_HASH','notes/geobc_matched20_solver_v78_plan.json',None),
        ('TS20','TS_ORIGINAL_BF16_GLOBAL_HASH','notes/geobc_matched20_solver_v78_plan.json',None)]:
        tp,ta,te,tb=terminal(planfile)
        if cp is None:
            ownname='FNO_GLOBAL_HASH20'if family=='FNO_GLOBAL_HASH'else'TS_CONTINUOUS20'
            cp=ta['arms'][ownname]['endpoints']['6000']['checkpoint_identity']
            order=tp['full_order_digest']
        else:order=tp['scientific']['full_order_digest']
        d.stat_matches(cp)
        assert tp['scientific']['scaler']==old['scaler']
        assert tp['scientific']['dataset_identity']['path']=='data/geobc_prospective_dataset_v1/manifest.json'
        candidates.append(dict(name=name,family=family,step=6000,seed=2026,training_plan_identity=d.identity(planfile,hashed=True),
            training_SCI=tp['scientific_digest'],terminal_binding=tb,checkpoint=cp,full_order_digest=order,training_boot_id=ta['execution']['boot_id']))
    sources={}
    for cad in cads:
        sources[cad]={}
        for component,ext in [('Field','.csv'),('FieldMesh','.h5'),('SolverDeck','.fem')]:
            file=f'data/raw_geobc_coverage/{component}/{cad}{ext}'
            receipt=d.read(f'data/raw_geobc_coverage/.receipts/{component}/{cad}{ext}.json')
            assert receipt['status']=='completed'
            ident=d.identity(file);ident['sha256']=receipt['observed_sha256']
            assert ident['bytes']==receipt['actual_bytes']
            sources[cad][component]=ident
    inherited=d.read('data/geobc_available_cad_pilot/manifest.json')['contract']
    contract=dict(raw_sources=sources,groups={'train_new160':[],'validation20':cads},scaler=old['scaler'],input_recipe=inherited['input_recipe'],
        scope='reserved2 target-free input math; no numeric targets before frozen3 Cartesian prediction barrier')
    contract_path='notes/geobc_reserve_holdout_v93_input_contract.json';d.write(contract_path,contract)
    science=dict(question='Check three already frozen same20-exposure solvers on the only2 locally complete, previously unused reserved CADs; no training/tuning and no finalwinner claim',
        CADs=cads,loads=['ver','hor','tor'],case_keys=d.keys(cads),candidates=candidates,raw_sources=sources,
        exposure_evidence_identity=d.identity(evidence_path,hashed=True),split_identity=d.identity('notes/geobc_prospective_cad_split_v1.json',hashed=True),
        input_contract_identity=d.identity(contract_path,hashed=True),dataset_identity=tp['scientific']['dataset_identity'],scaler=old['scaler'],
        GT_policy='All18 complete finite full-node encoded4+physical4 predictions immutable/hash/body revalidated before2 CSV known3 signedU/S parses; six label fields shared across all3 solvers',
        whole_campaign_budget_seconds=900,memory_limit_bytes=30500000000,GPU_lock='runs/geobc_exploration/.coordinator.lock',
        fixed_prediction_denominator=18,training_updates_allowed=0,scaler_fits_allowed=0,automatic_continuation=False,
        interpretation_limits=evidence['limits']+['Single2026seed matched20 exposure, different model capacities/precision and continuation history','New6 cannot replace old20CAD results or estimate architecture limits','No repeats/configuration changes based on this holdout'])
    design=dict(status='design_frozen_actual0',created_at_utc=d.utc(),scientific=science,scientific_digest=d.digest(science),actual_preparation=0,actual_evaluation=0)
    assert not d.path(DESIGN).exists();d.write(DESIGN,design)
    sr=d.read('notes/geobc_reserve_holdout_v93_source_review.json')
    assert sr['passed'] and sr['whole_source_reviewed'] and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    p=dict(kind='GeoBC_reserved2_frozen3_same20_prediction_first_v93',scientific=science,scientific_digest=d.digest(science),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/reserved2_frozen3_same20_v93')
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        source_review_identity=d.identity('notes/geobc_reserve_holdout_v93_source_review.json',hashed=True),
        upstream_terminal_bindings={x['name']:x['terminal_binding']for x in candidates})
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='metadata_prepared_release_pending',SCI=p['scientific_digest'],new_arrays=0,new_PT_bodies=0,new_GT=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","GT":0,"GPU":0}')
