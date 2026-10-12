"""Explicit target-free scientific I/O, invoked only by the released runner."""
from . import available_cad_pilot_data as d


def load_inputs(np,torch,manifest,keys,receipt,guard):
    from .prospective_dataset import resolve_case
    arrays={};specs={};cases={}
    for key in keys:
        record,items=resolve_case(manifest,key,False,100,True)
        assert set(items)=={'xyz','features','cloud','mask','grid','query_indices','query_weights'}
        assert manifest['case_roles'][key]=='DEV'
        n=record['num_nodes'];shapes={'xyz':(n,3),'features':(n,25),'cloud':(1024,3),'mask':(1024,),
            'grid':(29,64,64,64),'query_indices':(n,8),'query_weights':(n,8)}
        case=dict(key=key,scientific_group='DEV')
        for name,item in items.items():
            guard();file=item['path']
            if file not in arrays:
                d.stat_matches(item);receipt['input_array_attempts']+=1
                value=np.load(d.path(file),allow_pickle=False);receipt['input_array_reads']+=1
                assert value.shape==shapes[name]and str(value.dtype)in (('bool',)if name=='mask'else('int32','int64')if name=='query_indices'else('float32',))
                if value.dtype.kind=='f':assert np.isfinite(value).all()
                if name=='query_indices':assert value.min()>=0 and value.max()<64**3
                arrays[file]=torch.as_tensor(value,device='cuda');specs[file]=(name,item)
            else:assert specs[file]==(name,item)
            case[name]=arrays[file]
        case['cloud']=case['cloud'][None];case['mask']=case['mask'][None];case['grid']=case['grid'][None];cases[key]=case
    assert receipt['input_array_reads']==receipt['input_array_attempts']==len(arrays)
    return cases,arrays


def source_gate(np,science,receipt,guard):
    import time
    from . import elastic_pilot_data as parser,coverage_comparison_data as original
    force_vectors={};material_vectors={}
    for cad,item in science['raw_deck_identities'].items():
        guard();assert d.identity(item['path'],hashed=True)==item;receipt['source_deck_hash_reads']+=1
        receipt['source_deck_parse_attempts']+=1
        deck=parser._parse_deck(d.path(item['path']),science['deck_reader_policy'],receipt['execution']['guard']['entry_monotonic'],science['whole_budget_seconds'])
        receipt['source_deck_parses']+=1
        gen=parser._cards(d.path(item['path']),science['deck_reader_policy']['reader_scope'],receipt['execution']['guard']['entry_monotonic'],science['whole_budget_seconds'])
        try:controls=next(gen)[3]
        finally:gen.close()
        receipt['source_deck_header_reads']+=1
        mids={deck['psolid'][e['PID']]['MID']for e in deck['ctetra'].values()}
        if len(mids)!=1 or len(deck['materials'])!=1:raise parser.UnsupportedInput('One explicit MAT1 required')
        material=deck['materials'][next(iter(mids))];E=material['E'];nu=material['nu']
        if E is None or E<=0 or nu is None or not -1<nu<.5:raise parser.UnsupportedInput('Unsupported E/nu')
        if material['G']is not None and abs(material['G']/(E/(2*(1+nu)))-1)>1e-8:raise parser.UnsupportedInput('MAT1 G inconsistent')
        rho=parser._number(material['remaining_fields'][0])if material['remaining_fields']else None
        if rho is None or rho<=0:raise parser.UnsupportedInput('Explicit density required')
        material_vectors[cad]=[E/113800,nu,rho/4.47e-9];rows={};constraints=[]
        for load in ('ver','hor','tor'):
            selected=original._deck_case(deck,controls,load)
            if selected['selected_controls']['ANALYSIS'].upper()!='STATICS':raise parser.UnsupportedInput('Require active STATICS')
            if not selected['active_constraints']or any(x[2]!=0. for x in selected['active_constraints']):raise parser.UnsupportedInput('Missing/nonzero support motion')
            if any(x['kind']not in ('FORCE','MOMENT')for x in selected['active_actions']):raise parser.UnsupportedInput('Unsupported actions')
            f=np.zeros(3);mom=np.zeros(3)
            for action in selected['active_actions']:(f if action['kind']=='FORCE'else mom)[:]+=np.asarray(action['vector'])
            v=np.concatenate((f/42300,mom/565000));assert np.isfinite(v).all()and np.linalg.norm(v)>0
            force_vectors[cad+'/'+load]=v;cons=sorted((int(x[0]),''.join(sorted(x[1])),float(x[2]))for x in selected['active_constraints']);constraints.append(cons)
            rows[load]=dict(selected_controls=selected['selected_controls'],force_N=f.tolist(),moment_Nmm=mom.tolist(),constraints=cons)
        if not all(x==constraints[0]for x in constraints):raise parser.UnsupportedInput('Canonical supports differ')
        receipt['physics_gate'][cad]=dict(explicit_material=material,full_card_inventory=deck['cards'],zero_constraints=True,same_constraints_across3=True,STATICS_subcases=rows,ignored_original_controls=deck['ignored_controls'],PSOLID_controls={str(k):v['original_controls_not_replicated']for k,v in deck['psolid'].items()},conditional_fixedK_assumptions=True,exact_original_solver_fidelity_claimed=False)
        del deck;guard()
    assert len(receipt['physics_gate'])==20
    return force_vectors,material_vectors
