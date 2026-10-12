"""Authentic support reference GRID coordinates from an already accepted deck.

One target-free card stream adds the coordinates absent from cached physics
inputs. The old accepted source inventory remains binding, no K is rebuilt.
"""
def source_reference_coordinates(np,row,scope,a,guard):
    import time
    from . import available_cad_pilot_data as d
    from . import elastic_pilot_data as source
    from .elastic_pilot import ScientificNegative
    began=(time.time(),time.monotonic());item=row['raw_deck_coordinate_identity'];guard()
    if d.identity(item['path'],hashed=True)!=item:raise ScientificNegative('Authentic frozen deck identity changed')
    a['deck_coordinate_hash_reads']+=1;refs=row['source_support_constraints']['support_reference_GRID_IDs'];found={};rbe2={};inventory={};defaults=None;controls={};ignored=[]
    for count,(name,values,line,controls,ignored)in enumerate(source._cards(d.path(item['path']),scope,began[1],60)):
        if count%4096==0:guard()
        inventory[name]=inventory.get(name,0)+1;v=values+['']*16
        if name=='GRDSET':
            if defaults is not None or v[0]:raise ScientificNegative('Nonstandard GRDSET in accepted support coordinate stream')
            defaults={k:v[i]for k,i in (('CP',1),('CD',5),('PS',6),('SEID',7))}
        elif name=='GRID'and source._integer(v[0])in refs:
            ref=source._integer(v[0])
            if ref in found:raise ScientificNegative('Duplicate support reference GRID')
            found[ref]=dict(reference_GRID_ID=ref,raw_coordinates_mm=[source._number(x)for x in v[2:5]],
                CP=v[1],CD=v[5],PS=v[6],SEID=v[7],source_line=line)
        elif name=='RBE2':
            ref=source._integer(v[1]);cm=''.join(sorted(source._dofs(v[2])));members=[source._integer(x)for x in values[3:]if x]
            if ref in rbe2 or not members or len(set(members))!=len(members):raise ScientificNegative('Source support RBE2 member/reference ambiguity')
            rbe2[ref]=dict(reference_GRID_ID=ref,CM=cm,member_GRID_IDs=sorted(members),source_line=line)
    a['deck_coordinate_passes']+=1;guard();defaults=defaults or dict(CP='',CD='',PS='',SEID='')
    if sorted(found)!=refs or sorted(rbe2)!=refs or len(refs)!=4:raise ScientificNegative('All four authentic source supports are required')
    coordinates=[]
    for ref in refs:
        record=found[ref]
        for key in defaults:record[key]=record[key]or defaults[key]
        if any(source._integer(record[key],0)!=0 for key in ('CP','CD','SEID'))or source._dofs(record['PS'],True):raise ScientificNegative('Nonbasic/default constrained reference coordinate unsupported')
        xyz=record['raw_coordinates_mm']
        if None in xyz or not np.isfinite(xyz).all()or rbe2[ref]['CM']!='123456':raise ScientificNegative('Authentic GRID XYZ/RBE2 CM unsupported')
        coordinates.append(xyz)
    seconds=max(time.time()-began[0],time.monotonic()-began[1])
    if seconds>60:raise TimeoutError('PerCAD source coordinate pass budget')
    return np.asarray(coordinates,dtype=np.float64),dict(passed=True,reference_GRID_ID_order=refs,authentic_GRID_records=[found[k]for k in refs],
        actual_source_RBE2_records=[rbe2[k]for k in refs],full_card_inventory=inventory,source_case_controls=controls,source_ignored_header=ignored,
        coordinate_source='Exact original GRID CP0/CD0,not member centroid/output-row invention',source_identity=item,actual_source_hash_reads=1,
        actual_card_passes=1,source_coordinate_seconds=seconds,original_DEBUG_PSOLID_effects_unknown=True)

def validate_source_ports(t,inputs,row,source_facts):
    from .all_support_translation_pcg_v118 import validate_ports
    from .elastic_pilot import ScientificNegative
    ports,others,facts=validate_ports(t,inputs,dict(row,existing_port0_actual_facts=row['actual_support_ports']['ports'][0]))
    grid=inputs['grid_node_ids']
    for j,record in enumerate(source_facts['actual_source_RBE2_records']):
        actual=grid[inputs['region_support_'+str(j)]].sort().values.cpu().tolist()
        if actual!=record['member_GRID_IDs']:raise ScientificNegative('Actual cached source region differs from authentic RBE2 members')
    facts['source_original_rotations_prescribed_zero']=facts.pop('rotations_prescribed_zero')
    facts['new_rotation_capability_requires_independent_verification']=True
    return ports,others,facts

def make_lift(t,inputs,ports,others,reference_xyz,motions):
    from .support_rigid_motion_response_v120 import rigid_support_lift
    lift=rigid_support_lift(t,inputs['vertices_mm'],ports,reference_xyz,motions)
    assert bool((lift[:,inputs['free_node_ids']]==0).all()and (lift[:,others]==0).all())
    # The position of a reference may be absent from the H5 rows. Only a real
    # existing master row could be assigned, through the validated port list.
    for j,rows in enumerate(ports):
        delta=inputs['vertices_mm'][rows]-reference_xyz[j]
        wanted=motions[:,j,None,:3]+t.cross(motions[:,j,None,3:].expand(-1,len(rows),-1),delta[None].expand(3,-1,-1),dim=-1)
        assert t.equal(lift[:,rows],wanted)
    return lift
