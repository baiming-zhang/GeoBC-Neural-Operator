"""Located continuous U/S response for registered canonical load cases."""
import torch
from .kinematic_query import KinematicCovariateQuery
from .locator import PhysicalULocator
from .query_stencil import stencil
from .physical_loss import decode


class LocatedNeuralResponse:
    def __init__(self,model,vertices,cells,covariates,reference_load6,cases,scale,guard):
        if model.input_mode not in {'ACTUAL','U_ONLY','GRAD_ONLY'} or set(cases)!={'ver','hor','tor'}:
            raise ValueError('Registered physical-input canonical cases required')
        self.model,self.cases,self.scale,self.guard=model,cases,scale,guard
        self.covariates=KinematicCovariateQuery(torch,vertices,cells,covariates,reference_load6,guard)
        self.locator=PhysicalULocator(torch,vertices,cells,guard)
        self.context={}
        model.eval()
        with torch.no_grad():
            for load,case in cases.items():
                self.context[load]=model.encode_case(case)
                guard()

    def prepare_query(self,xyz_mm,cell_rows,load):
        if load not in self.cases:
            raise ValueError('Load is not registered')
        self.guard()
        N,_,_,_=self.covariates.reference.invert(xyz_mm,cell_rows)
        nodes=self.covariates.cells[cell_rows]
        case=self.cases[load]; index=['ver','hor','tor'].index(load)
        xyz=torch.einsum('qi,qia->qa',N,case['xyz'][nodes].double()).float()
        native=torch.einsum('qi,qia->qa',N,case['features'][nodes].double()).float()
        cov=torch.einsum('qi,qia->qa',N,self.covariates.inputs12[index,nodes]).float()
        indices,weights=stencil(xyz)
        return dict(xyz=xyz,features=native,kinematic_inputs=cov,query_indices=indices,query_weights=weights)

    def query_in_cells(self,xyz_mm,cell_rows,load):
        query=self.prepare_query(xyz_mm,cell_rows,load)
        context,latent=self.context[load]
        physical=decode(self.model.query(query,0,len(xyz_mm),context,latent),self.scale)
        if not bool(torch.isfinite(physical).all()):
            raise FloatingPointError('Nonfinite continuous U/S query')
        return physical

    def query(self,xyz_mm,load):
        if load not in self.cases:
            raise ValueError('Load is not registered')
        with torch.no_grad():
            ids,records=self.locator.locate(xyz_mm.detach())
        if not bool((ids>=0).all()) or any(row['status']=='ambiguous_negative' for row in records):
            return dict(status='rejected',physical=None,cell_rows=ids,location=records)
        return dict(status='completed',physical=self.query_in_cells(xyz_mm,ids,load),cell_rows=ids,location=records)


class LocatedFieldwiseNeuralResponse(LocatedNeuralResponse):
    """Route U from a displacement-input model and S from a gradient-input model.

    Both networks share point location, interpolated inputs and the output scaler.
    Loads remain the registered ver/hor/tor cases.
    """
    def __init__(self,u_model,s_model,vertices,cells,covariates,reference_load6,cases,scale,guard):
        if u_model.input_mode!='U_ONLY' or s_model.input_mode!='GRAD_ONLY':
            raise ValueError('U_ONLY and GRAD_ONLY models required')
        super().__init__(u_model,vertices,cells,covariates,reference_load6,cases,scale,guard)
        self.s_model=s_model
        self.s_context={}
        s_model.eval()
        with torch.no_grad():
            for load,case in cases.items():
                self.s_context[load]=s_model.encode_case(case)
                guard()

    def query_in_cells(self,xyz_mm,cell_rows,load):
        query=self.prepare_query(xyz_mm,cell_rows,load)
        context,latent=self.context[load]
        u=decode(self.model.query(query,0,len(xyz_mm),context,latent),self.scale)[:,:3]
        context,latent=self.s_context[load]
        s=decode(self.s_model.query(query,0,len(xyz_mm),context,latent),self.scale)[:,3:4]
        physical=torch.cat((u,s),dim=1)
        if not bool(torch.isfinite(physical).all()):
            raise FloatingPointError('Nonfinite fieldwise response')
        return physical
