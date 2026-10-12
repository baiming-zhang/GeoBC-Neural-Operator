"""Six independently solved physical displacement bases at one RBE reference."""
from .physical_U_query_v104 import PhysicalUQuery

class FullRank6PhysicalUQuery(PhysicalUQuery):
    def __init__(self,torch,vertices,cells,basis,load6,guard):
        self.torch,self.vertices,self.cells,self.basis,self.B,self.guard=torch,vertices,cells,basis,load6,guard
        t=torch;guard()
        assert vertices.is_cuda and cells.is_cuda and basis.is_cuda and load6.is_cuda
        assert vertices.dtype==basis.dtype==load6.dtype==t.float64 and cells.dtype==t.int64
        assert vertices.ndim==2 and vertices.shape[1]==3 and cells.ndim==2 and cells.shape[1]==10
        assert basis.shape==(6,len(vertices),3) and load6.shape==(6,6)
        assert bool(t.isfinite(vertices).all() and t.isfinite(basis).all() and t.isfinite(load6).all())
        assert bool(((cells>=0)&(cells<len(vertices))).all())
        rank=int(t.linalg.matrix_rank(load6));condition=float(t.linalg.cond(load6))
        if rank!=6 or condition>1e12:raise ValueError('Six physical load directions are not independent')
        self.metadata=dict(actual_allN=len(vertices),actual_cells=len(cells),physical_load_rank=rank,
            physical_load_condition=condition,physical_U_units='mm',force_units='N',moment_units='Nmm',
            scope='same_geometry_material_constraints_RBEreference',new_OptiStruct_GT_accuracy=False,C1_claim=False)
    def coefficients(self,load6):
        t=self.torch;self.guard()
        assert load6.dtype==t.float64 and load6.ndim==2 and load6.shape[1]==6 and bool(t.isfinite(load6).all())
        return t.linalg.solve(self.B,load6.T).T
    def query(self,xyz,element_rows,load6):
        values=self.basis_at(xyz,element_rows);c=self.coefficients(load6)
        assert c.shape==(len(xyz),6)
        return self.torch.einsum('qk,qka->qa',c,values)
    def query_coefficients(self,xyz,element_rows,c):
        assert c.shape==(len(xyz),6) and c.dtype==self.torch.float64 and bool(self.torch.isfinite(c).all())
        return self.query(xyz,element_rows,c@self.B.T)
