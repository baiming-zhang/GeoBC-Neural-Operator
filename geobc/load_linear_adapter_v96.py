"""Reusable physical-U superposition adapter over three real canonical loads.

No job/data/PT/label I/O. The released caller must prove common geometry,
material, homogeneous supports and canonical load space. Raw signedS remains
the original network response; no scalar-stress superposition is assumed.
This module is a SOURCE prototype until the independent real-N qualification.
"""
import torch
from .macro_training import encode_case,query_case
from .physical_field_loss import decode_physical


class LoadLinearDisplacementAdapter:
    def __init__(self,model,canonical_cases,physical_scale,guard=lambda:None):
        if len(canonical_cases)!=3:raise ValueError('Exactly three actual canonical load inputs')
        self.model=model;self.cases=list(canonical_cases);self.scale=physical_scale;self.guard=guard
        first=self.cases[0]
        for case in self.cases:
            for key in ('xyz','cloud','mask','query_indices','query_weights'):
                if not torch.equal(first[key],case[key]):raise ValueError('Canonical geometry/stencil differs')
            if not torch.equal(first['features'][:,:7],case['features'][:,:7])or not torch.equal(first['features'][:,13:],case['features'][:,13:]):raise ValueError('Canonical support/material/geometric query inputs differ')
            # Existing independent FP32 rasterizations may differ by atomics
            # roundoff. Keep every original basis input rather than replacing
            # it; the caller records the observed non-load difference.
            if not torch.allclose(first['grid'][:,:11],case['grid'][:,:11],atol=1e-4,rtol=0)or not torch.allclose(first['grid'][:,17:],case['grid'][:,17:],atol=1e-4,rtol=0):raise ValueError('Canonical grid non-load inputs differ')
            vector=case['features'][0,7:13]
            if not torch.equal(case['features'][:,7:13],vector[None].expand(len(case['xyz']),-1)):raise ValueError('A global uniform normalized force/moment vector is required')
        self.basis=torch.stack([c['features'][0,7:13]for c in self.cases],1).to(torch.float64)
        if int(torch.linalg.matrix_rank(self.basis))!=3:raise ValueError('Canonical force/moment basis lacks rank3')
        self.pseudoinverse=torch.linalg.pinv(self.basis)
        self.condition_number=float(torch.linalg.cond(self.basis))
        self.prepared=None

    def coefficients(self,normalized_force6):
        if normalized_force6.shape!=(6,)or not torch.isfinite(normalized_force6).all():raise ValueError('Finite6 normalized global force/moment values required')
        vector=normalized_force6.to(dtype=torch.float64,device=self.basis.device)
        coeff=self.pseudoinverse@vector
        residual=self.basis@coeff-vector
        if float(residual.abs().max())>2e-6:raise ValueError('Requested load lies outside the verified canonical rank3 subspace')
        return coeff

    def prepare(self):
        if self.prepared is not None:raise RuntimeError('One canonical encoding per adapter; no automatic replay')
        self.prepared=[]
        for case in self.cases:
            self.guard();self.prepared.append(encode_case(self.model,case))
        return self

    def query(self,mixed_query,start,stop,normalized_force6,raw_context,raw_latent):
        """Return physical4 plus encoded4 with untouched original encodedS.

        mixed_query may be actual continuous spatial queries. Its analytic
        non-load features and stencil belong to this verified geometry. Raw
        context/latent are the original model's encoding of the requested load.
        All three base queries use identical coordinates/non-load features.
        """
        if self.prepared is None:raise RuntimeError('Canonical encodings not prepared')
        self.guard();coeff=self.coefficients(normalized_force6)
        if not torch.allclose(mixed_query['features'][start:stop,7:13],normalized_force6.to(mixed_query['features'])[None].expand(stop-start,-1),atol=2e-6,rtol=0):raise ValueError('Mixed query and requested global force disagree')
        raw=query_case(self.model,mixed_query,start,stop,raw_context,raw_latent).to(torch.float32)
        raw_physical=decode_physical(raw,self.scale);u=torch.zeros_like(raw_physical[:,:3])
        for i,(context,latent)in enumerate(self.prepared):
            self.guard();basis_query={name:mixed_query[name][start:stop]for name in ('xyz','query_indices','query_weights')}
            features=mixed_query['features'][start:stop].clone();features[:,7:13]=self.basis[:,i].to(features.dtype)[None]
            basis_query['features']=features
            base_encoded=query_case(self.model,basis_query,0,stop-start,context,latent).to(torch.float32)
            u=u+coeff[i]*decode_physical(base_encoded,self.scale)[:,:3]
        physical=torch.cat((u,raw_physical[:,3:]),1)
        # U conversion only: preserve the exact raw encodedS bits instead of
        # decoding/re-encoding an unknown signed-scalar convention.
        mean=u.new_tensor(self.scale.mean[:3]);std=u.new_tensor(self.scale.std[:3])
        encoded=torch.cat((((u-mean)/std).to(torch.float32),raw[:,3:]),1)
        if not torch.isfinite(physical).all()or not torch.isfinite(encoded).all():raise FloatingPointError('Nonfinite adapter full query')
        if not torch.equal(encoded[:,3],raw[:,3]):raise RuntimeError('Raw signedS channel changed')
        self.guard();return physical,encoded
