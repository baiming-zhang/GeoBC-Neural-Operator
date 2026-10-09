"""Canonical BC roles on the original fixed boundary cloud, without targets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .conditioned_field import GeometryCloud,load_geometry_cloud
from .displacement_condition import file_identity,host_path
from .local_geometry import xyz_sha256

ROLE_ORDER = ("support_union_original_DOF0_to5", "load_member_original_feature6")


def _sha(values): return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class BCRoleCloud(GeometryCloud):
    metadata: dict

    @property
    def roles(self): return self.points[:,3:5]
    @property
    def xyz_points(self): return self.points[:,:3]

    def record(self):
        result = super().record()
        result.update(self.metadata)
        result.update(global_cloud_input_dimension=5,role_order=list(ROLE_ORDER),
                      points5_sha256=_sha(self.points),coordinates_sha256=_sha(self.xyz_points),
                      roles_sha256=_sha(self.roles),mask_sha256=_sha(self.mask),node_indices_sha256=_sha(self.node_indices),
                      role_support_selected_nodes=int(self.roles[self.mask,0].sum()),
                      role_load_selected_nodes=int(self.roles[self.mask,1].sum()),
                      role_support_and_load_selected_nodes=int(np.logical_and(self.roles[self.mask,0],self.roles[self.mask,1]).sum()),
                      role_policy="supportUnion=any originalfeatures0:6; loadMember=originalfeatures6; canonical ver only, same existing sampled indices",
                      geometry_target_independent=True,geometry_load_independent=True,targets_read=False,
                      role_members_read=["features (only0:7 used)","xyz","sample_key","label_alignment"])
        return result


def load_bc_role_cloud(path,*,cad_key=None,seed=2026,cloud_points=1024):
    """Reuse frozen loader's coords/mask/indices without resampling/reordering.

    Read canonical ver input roles only. No displacement/stress member is
    accessed, and no target statistics or load magnitudes enter the cloud.
    Sampling is exactly the original CAD-key stable boundary-node sampler.
    """
    path = host_path(path)
    base = load_geometry_cloud(path,cloud_points=cloud_points,seed=seed,cad_key=cad_key)
    with np.load(path,allow_pickle=False) as stored:
        key = str(stored["sample_key"].item())
        xyz = np.asarray(stored["xyz"],np.float32)
        features = np.asarray(stored["features"],np.float32)
        alignment = str(stored["label_alignment"].item())
    if key!=f"{base.cad_key}/ver" or alignment!="official CSV GRID coordinates -> H5 vertices":
        raise ValueError("BC-role cloud requires the original GRID-aligned canonical ver CAD input")
    if features.shape!=(len(xyz),25) or xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(features[:,:7]).all() or not np.isin(features[:,:7],[0,1]).all():
        raise ValueError("Original25 binary support DOF0:6/loadMember6 role inputs are required")
    valid = base.mask
    if not np.array_equal(base.points[valid],xyz[base.node_indices[valid]]) or not np.all(base.node_indices[~valid]==-1):
        raise ValueError("Fixed geometry cloud coordinates/node indices differ from role-source original order")
    roles = np.zeros((len(base.points),2),np.float32)
    selected = features[base.node_indices[valid],:7]
    roles[valid,0] = np.any(selected[:,:6]!=0,axis=1)
    roles[valid,1] = selected[:,6]
    points = np.concatenate((base.points,roles),axis=-1)
    metadata = {"canonical_sample_key":key,"role_source":file_identity(path),"original_node_count":len(xyz),
                "source_xyz_sha256":xyz_sha256(xyz),"label_alignment":alignment}
    return BCRoleCloud(points,base.mask,base.node_indices,base.cad_key,base.source,base.pool,base.candidate_nodes,base.seed,metadata)


__all__ = ["BCRoleCloud","load_bc_role_cloud","ROLE_ORDER"]
