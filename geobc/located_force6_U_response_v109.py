"""Reusable physicalXYZ + reference force6 -> U(mm), without caller cell IDs.

The discrete location branch is held fixed for autograd; the accepted curved
inverse and six-basis physical response remain on the differentiable graph.
"""

class LocatedForce6UResponse:
    def __init__(self,api,locator,source_contract):
        assert api.vertices is locator.x and api.cells is locator.cells
        assert api.metadata['physical_load_rank']==6
        assert source_contract['REFC']=='123456' and source_contract['material_constraints_fixed']
        assert source_contract['physical_U_units']=='mm'
        assert source_contract['force_units']=='N' and source_contract['moment_units']=='Nmm'
        self.api,self.locator,self.source_contract=api,locator,dict(source_contract)
        self.metadata=dict(interface='physicalXYZ_referenceForce6_to_Umm',caller_element_IDs_required=False,
            physical_load_rank=6,physical_reference_GRID_ID=source_contract['reference_GRID_ID'],
            fixed_geometry_material_constraints=True,location_branch_discrete=True,C1_claim=False,
            new_OptiStruct_GT_accuracy=False,signed_S_provided=False)

    def query(self,xyz,force6):
        torch=self.api.torch
        assert xyz.ndim==2 and xyz.shape[1]==3 and force6.shape==(len(xyz),6)
        # Invalid loads fail before any location; no projection or mean floors.
        self.api.coefficients(force6)
        with torch.no_grad():
            ids,records=self.locator.locate(xyz.detach())
        if not bool((ids>=0).all()) or any(row['status']=='ambiguous_negative'for row in records):
            return dict(status='rejected',physical_U=None,located_cell_rows=ids,location_records=records,
                        complete_requested_point_count=len(xyz),subset_prediction_allowed=False)
        values=self.api.query(xyz,ids,force6)
        assert bool(torch.isfinite(values).all())
        return dict(status='completed',physical_U=values,located_cell_rows=ids,location_records=records,
                    complete_requested_point_count=len(xyz),subset_prediction_allowed=False)
