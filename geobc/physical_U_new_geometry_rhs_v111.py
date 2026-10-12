"""Six reference RHS on a newly parsed geometry, preserving actual RBE transfer.

This is SOURCE only until a separately released new-geometry campaign executes
it. Canonical loads come from actual expanded source LOAD actions; the three
additional reference directions have frozen amplitudes, not fitted labels.
"""

def build_reference_rhs(t,inputs,canonical_rhs,canonical_columns,force_amplitude,moment_amplitude,guard,tolerance):
    from .elastic_pilot import ScientificNegative
    guard();x=inputs['vertices_mm'];members=inputs['rbe3_member_node_ids'];H=inputs['rbe3_Hbar']
    length=float(inputs['rbe3_length_mm']);offset=x[members]-inputs['reference_xyz_mm']
    assert x.is_cuda and x.dtype==t.float64 and members.dtype==t.int64
    assert length>0 and force_amplitude>0 and moment_amplitude>0
    first=t.as_tensor(canonical_columns,dtype=t.float64,device=x.device).T.contiguous()
    assert first.shape==(6,3) and bool(t.isfinite(first).all())
    added=t.tensor([[0.,force_amplitude,0.,0.,0.,0.],[0.,0.,0.,moment_amplitude,0.,0.],
                    [0.,0.,0.,0.,moment_amplitude,0.]],dtype=t.float64,device=x.device).T
    B=t.cat((first,added),dim=1)
    rank=int(t.linalg.matrix_rank(B));condition=float(t.linalg.cond(B))
    if rank!=6 or not condition<=1e12:raise ScientificNegative('Source reference six directions not full rank')
    assert canonical_rhs.shape==(3,len(x),3) and canonical_rhs.dtype==t.float64
    forces=[];records=[]
    for j in range(6):
        guard();f=B[:,j];scaled=t.cat((f[:3],f[3:]/length));member_force=(H.T@scaled).reshape(-1,3)
        force=t.zeros_like(x);force.index_add_(0,members,member_force)
        ferr=float((member_force.sum(0)-f[:3]).abs().max())/max(1.,float(scaled.norm()))
        merr=float((t.cross(offset,member_force,dim=-1).sum(0)-f[3:]).abs().max())/max(1.,length*float(scaled.norm()))
        witness=t.arange(member_force.numel(),dtype=t.float64,device=x.device).sin()
        work=float((witness@(H.T@scaled)-(H@witness)@scaled).abs())/max(1.,float(witness.norm())*float((H.T@scaled).norm()))
        if max(ferr,merr,work)>tolerance:raise ScientificNegative('New geometry RBE force/moment/virtual-work failure')
        if j<3 and not t.allclose(force,canonical_rhs[j],rtol=1e-12,atol=1e-12):
            raise ScientificNegative('Source expanded canonical RHS differs from six-reference construction')
        assert bool(t.isfinite(force).all())
        forces.append(force);records.append(dict(direction=j,force_relative=ferr,moment_relative=merr,virtual_work_relative=work,passed=True))
    return B,t.stack(forces),dict(rank=rank,condition=condition,six_transfer_records=records,original_member_positions_used=True)
