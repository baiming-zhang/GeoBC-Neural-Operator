"""Target/load-independent fixed shape cloud frame; no IO on import."""


def canonicalize(np, points, mask):
    x = np.asarray(points, dtype=np.float64); valid = np.asarray(mask, dtype=bool)
    assert x.shape == (1024, 3) and valid.shape == (1024,) and valid.sum() >= 4 and np.isfinite(x).all()
    mean = x[valid].mean(0); centered = x[valid]-mean
    covariance = centered.T@centered/len(centered)
    values, vectors = np.linalg.eigh(covariance); values = values[::-1]; vectors = vectors[:, ::-1].copy()
    rms = float(np.sqrt(np.square(centered).sum()/len(centered)))
    if not rms > 0: raise ValueError('Positive geometry RMS required')
    gaps = values[:-1]-values[1:]
    degenerate = bool((gaps <= 1e-6*values[0]).any())
    indices = []
    if degenerate:
        rotation = np.eye(3)
    else:
        rotation = vectors
        for axis in (0, 1):
            projection = centered@rotation[:, axis]; index = int(np.argmax(np.abs(projection))); indices.append(index)
            if projection[index] < 0: rotation[:, axis] *= -1
        rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
    assert np.max(np.abs(rotation.T@rotation-np.eye(3))) < 1e-10 and abs(np.linalg.det(rotation)-1) < 1e-10
    result = np.zeros_like(x, dtype=np.float32); result[valid] = (centered@rotation/rms).astype(np.float32)
    assert np.isfinite(result).all() and np.all(result[~valid] == 0)
    reconstructed = result[valid].astype(np.float64)*rms@rotation.T+mean
    error = float(np.max(np.abs(reconstructed-x[valid]))); assert error < 2e-6
    return result, dict(centroid=mean.tolist(), rotation=rotation.tolist(), RMS_radius=rms, eigenvalues=values.tolist(),
        eigenvalue_gaps=gaps.tolist(), degenerate_identity_fallback=degenerate, sign_anchor_valid_indices=indices,
        valid_nodes=int(valid.sum()), proper_orthogonal_frame=True, max_roundtrip_normalized_error=error,
        canonical_valid_RMS=float(np.sqrt(np.square(result[valid]).sum()/valid.sum())), GT_BC_values_used=False)


def apply_to_cache(cache, np, torch, guard):
    from .fno_stress_specialist_v54 import tree_digest
    records = {}; transformed = {}; origins = {}; invariant_fields = ('xyz', 'features', 'mask', 'grid', 'query_indices', 'query_weights')
    for key in cache.keys:
        guard(); case = cache.case(key); cad = key.split('/')[0]
        before = {name: (id(case[name]), case[name].data_ptr(), case[name]._version) for name in invariant_fields}
        origin = (cache.case_specs[key]['cloud'], cache.case_specs[key]['mask'])
        if cad not in records:
            points = case['cloud'].detach().cpu().numpy().reshape(1024, 3); mask = case['mask'].detach().cpu().numpy().reshape(1024)
            value, record = canonicalize(np, points, mask)
            tensor = torch.as_tensor(value, device=case['cloud'].device).reshape(1, 1024, 3)
            records[cad] = dict(**record, original_cloud_identity=cache.specs[origin[0]]['identity'],
                original_mask_identity=cache.specs[origin[1]]['identity'], original_cloud_tensor_digest=tree_digest(case['cloud']),
                canonical_cloud_tensor_digest=tree_digest(tensor), shared_loads=[])
            transformed[cad] = tensor; origins[cad] = origin
            cache.receipt['actual_tensor_bytes'] += tensor.numel()*tensor.element_size()
        assert origin == origins[cad]
        case['cloud'] = transformed[cad]; records[cad]['shared_loads'].append(key.split('/')[1])
        assert before == {name: (id(case[name]), case[name].data_ptr(), case[name]._version) for name in invariant_fields}
    assert len(records) == 3 and all(set(z['shared_loads']) == {'ver', 'hor', 'tor'} for z in records.values())
    return dict(CAD_transform_computations=3, original_cloud_arrays_reopened=0, frames=records,
        global_input_tensor_references_and_versions_unchanged=True, labels_not_accessed_or_transformed=True,
        no_learned_context_cached=True, coordinate_domain='Geometry encoder cloud only; original global query/grid/BC retained')
