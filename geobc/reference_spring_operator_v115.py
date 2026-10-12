"""Physical grounded reference spring, with scalar and three-lane fine actions."""
class ScalarReferenceSpring:
    def __init__(self,t,fine,H,members,diagonal,guard):
        self.t,self.fine,self.H,self.members,self.diagonal,self.guard=t,fine,H,members,diagonal,guard
        self.calls=0;self.node_count=fine.node_count
        assert H.dtype==diagonal.dtype==t.float64 and members.dtype==t.int64
        assert H.shape==(6,3*len(members))and diagonal.shape==(6,)and bool((diagonal>0).all())
    def spring_action(self,u):
        t=self.t;self.guard();assert u.shape==(self.node_count,3)
        q=self.H@u[self.members].reshape(-1)
        member=(self.H.T@(self.diagonal*q)).reshape(-1,3)
        out=t.zeros_like(u);out.index_add_(0,self.members,member);return out
    def apply(self,u):
        self.calls+=1;return self.fine.apply(u)+self.spring_action(u)

class ThreeLaneReferenceSpring:
    def __init__(self,t,fine_parallel,scalar,guard):
        self.t,self.fine,self.original,self.guard=t,fine_parallel,scalar,guard;self.calls=0
    def apply(self,u):
        t=self.t;s=self.original;self.guard();assert u.shape==(3,s.node_count,3)
        q=u[:,s.members].reshape(3,-1)@s.H.T
        member=((q*s.diagonal)@s.H).reshape(3,-1,3)
        out=self.fine.apply(u);out.index_add_(1,s.members,member);self.calls+=1;return out

def qualify(t,scalar,parallel,free):
    # Geometry/DOF-order fixtures; no basis values, GT or validation errors.
    n=scalar.node_count;index=t.arange(free.numel()*3,dtype=t.float64,device='cuda').reshape(-1,3)
    u=t.zeros((n,3),dtype=t.float64,device='cuda');v=t.zeros_like(u)
    u[free]=index.sin();v[free]=(index*.37+.2).cos()
    lanes=t.stack((u,v,u+v));actual=parallel.apply(lanes)
    reference=t.stack([scalar.apply(x)for x in lanes])
    errors=t.linalg.vector_norm(actual-reference,dim=(1,2))/t.linalg.vector_norm(reference,dim=(1,2))
    action_u,action_v=actual[0],actual[1];uv=(u*action_v).sum();vu=(v*action_u).sum()
    symmetry=float((uv-vu).abs()/t.sqrt((u*action_u).sum()*(v*action_v).sum()))
    energies=[float((x*scalar.spring_action(x)).sum())for x in (u,v)]
    assert bool(t.isfinite(actual).all())and bool((errors<=1e-9).all())and symmetry<=1e-9 and min(energies)>0
    return dict(passed=True,actual_allN=n,actual_RHS_lanes=3,scalar_parity_relative=errors.cpu().tolist(),
        modified_K_symmetry_relative=symmetry,spring_positive_energies=energies,fixtures=2,
        inherited_eight_fine_K_guards_repeated=0,physical_spring_not_numerical_shift=True)
