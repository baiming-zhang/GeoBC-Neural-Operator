"""Produce three reviewed descriptors from metadata only; never array bodies."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.physical_U_response_package_contract_v110 import inputs,PLAN,REVIEW,SOURCE_REVIEW,DESCRIPTORS,OUTPUT

def prepare():
    design,b=inputs();s=design['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==v for k,v in sr['source_pins'].items())
    assert not any(d.path(k).exists()for k in (PLAN,REVIEW,DESCRIPTORS))
    packages={}
    for cad,row in s['cases'].items():
        records={}
        for name in ('ver','hor','tor','Fy','Mx','My'):
            identity=row['basis_manifest_identities'][name];rec=d.read(identity['path'])
            records[name]=dict(manifest_identity=identity,physical_U=rec['physical_U'])
        scope=dict(REFC=row['source_transfer_facts']['REFC'],reference_GRID_ID=row['source_transfer_facts']['reference_GRID_ID'],
            material_constraints_fixed=True,physical_U_units='mm',force_units='N',moment_units='Nmm')
        payload=dict(kind='independent_approximate_P2_physical_U_force6',CAD=cad,basis_order=['ver','hor','tor','Fy','Mx','My'],
            geometry_identity=row['input_cache_identity'],geometry_schema=row['expected_geometry'],basis_records=records,
            reference_load6_columns=row['reference_load6_columns'],scope=scope,
            source_provenance=dict(source_transfer_facts=row['source_transfer_facts'],accepted_prior_terminal=b,
                design_SCI=design['scientific_digest'],recipe='curvedP2_fourQ_weightedLS_RBE3;originalDEBUG_PSOLID_unknown'),
            inherited_cost=dict(original_basis=row['inherited_actual_basis_cost'],missing_basis_seconds=row['missing_basis_cost_seconds'],
                independent_validation_seconds=row['independent_validation_cost_seconds'],freshgeometry_fullcold_seconds=None),
            new_OptiStruct_GT_accuracy=False,signedS_provided=False)
        packages[cad]=dict(schema='GeoBC_physical_U_response_package_v1',payload=payload,payload_digest=d.digest(payload))
    d.write(DESCRIPTORS,dict(status='metadata_only_review_pending',packages=packages,scientific_body_reads=0))
    science=dict(s,implementation=dict(descriptor_identity=d.identity(DESCRIPTORS,hashed=True),
        reviewed_payload_digests={cad:pkg['payload_digest']for cad,pkg in packages.items()},
        tamper_fixtures=['physical_U_unit','field_identity','basis_order'],tamper_expected_digest='original_reviewed_digest_unchanged',
        package_timing='One actual factorycall perCAD;instrument NumPy body IO and locator constructor without altering accepted loader bytes',
        direct_reference_after_complete_barrier=True,numpy_dtype_namespace='torch.float64/float32/int64 to float64/float32/int64;no cast'))
    p=dict(kind='GeoBC_physical_U_response_package_v110',scientific=science,scientific_digest=d.digest(science),
        source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],
        source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest'],descriptor_count=3))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
