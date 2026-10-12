"""Metadata contract and signed-stress error identities; no IO on import."""
import math
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity

UPSTREAM = 'notes/geobc_solver_seed_readout_v83_plan.json'
DESIGN = 'notes/geobc_signed_stress_bottleneck_v84_design.json'
PLAN = 'notes/geobc_signed_stress_bottleneck_v84_plan.json'
REVIEW = 'notes/geobc_signed_stress_bottleneck_v84_root_review.json'
SOURCE_REVIEW = 'notes/geobc_signed_stress_bottleneck_v84_source_review.json'
OUTPUT = 'runs/geobc_exploration/signed_stress_bottleneck_v84'
SOURCES = ('geobc/signed_stress_diagnostic_v84.py', 'tools/run_geobc_signed_stress_v84.py',
           'tools/prepare_geobc_signed_stress_v84.py')
DEFINITIONS = dict(dtype='float64', hotspot='same top1percent absolute truth nodes as existing physical_metrics',
    nearzero='abs(trueS) <= .01 * RMS(trueS), exact0 separately retained',
    exact_identity='signed_SSE = magnitude_only_SSE + 4 sum(abs(predS * trueS) on opposite nonzero signs)',
    averaging='equal-case mean, full fixed60 per arm; two new paired seeds per package, not3seed or120independentCAD',
    null_policy='Zero denominator/undefined correlation remains null; no epsilon or incomplete-subset mean',
    oracle_policy='Truth-conditioned diagnostics only, no deployable correction or actual solver accuracy claim')


def inputs():
    p, a, e, b = terminal(UPSTREAM); design = d.read(DESIGN)
    assert d.digest(design['scientific']) == design['scientific_digest']
    assert a['protocol_qualified'] and a['prediction_barrier_complete'] and a['committed_finite_predictions'] == 240
    assert a['owned_GT_array_reads'] == 60 and a['checkpoint_reads'] == 4
    report = d.read('results/geobc_solver_seed_readout_v83/manifest.json')
    decision = d.read('notes/geobc_solver_seed_readout_v83_actual_decision.json')
    assert report['status'] == 'completed' and report['terminal_binding'] == b and decision['PNG_visually_accepted']
    keys = design['scientific']['case_keys']; names = design['scientific']['candidate_names']
    assert keys == p['scientific']['case_keys'] and names == [x['name'] for x in p['scientific']['candidates']]
    assert len(keys) == 60 and len(names) == 4
    barrier = d.read(d.path(p['output'])/'prediction_barrier.json')
    assert barrier['all_cartesian_entries'] and barrier['finite_predictions'] == 240 and barrier['inventory'] == a['prediction_inventory']
    truth = d.read(check_identity(p['scientific']['GT_manifest_identity']))
    assert set(truth['records']) == set(keys)
    return dict(plan=p, summary=a, external=e, binding=b, design=design, truth=truth,
        barrier_identity=d.identity(d.path(p['output'])/'prediction_barrier.json', hashed=True))


def science_for(up):
    return dict(design=up['design']['scientific'], design_SCI=up['design']['scientific_digest'],
        definitions=DEFINITIONS, upstream_SCI=up['plan']['scientific_digest'], upstream_binding=up['binding'],
        barrier_identity=up['barrier_identity'], GT_manifest_identity=up['plan']['scientific']['GT_manifest_identity'],
        GPU_lock=up['plan']['scientific']['GPU_lock'])


def approved(plan_path):
    p = d.read(plan_path); r = d.read(p['root_review']); up = inputs()
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['kind'] == 'GeoBC_signedS_bottleneck_CPU_v84' and p['output'] == OUTPUT
    assert p['scientific'] == science_for(up) and d.digest(p['scientific']) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert r['upstream_terminal_binding'] == up['binding']
    return p, r, up


def ratio(a, b):
    return a/b if b > 0 else None


def decompose(np, truth, prediction):
    y = np.asarray(truth, dtype=np.float64); v = np.asarray(prediction, dtype=np.float64)
    assert y.ndim == 1 and v.shape == y.shape and len(y) > 0
    assert np.isfinite(y).all() and np.isfinite(v).all()
    n = len(y); y2 = float(np.square(y).sum()); p2 = float(np.square(v).sum())
    hotn = max(1, math.ceil(.01*n)); hot = np.argpartition(np.abs(y), n-hotn)[-hotn:]
    err = v-y; signed = float(np.square(err).sum())
    magnitude = float(np.square(np.abs(v)-np.abs(y)).sum())
    opposite = y*v < 0; penalty = 4*float(np.abs(y[opposite]*v[opposite]).sum())
    assert math.isclose(signed, magnitude+penalty, rel_tol=1e-10, abs_tol=1e-12)
    zero = y == 0; near = np.abs(y) <= .01*math.sqrt(y2/n)
    sign_oracle = np.abs(v)*np.sign(y); magnitude_oracle = np.abs(y)*np.sign(v)
    def rel(pred, index=None):
        yy, pp = (y, pred) if index is None else (y[index], pred[index])
        value = ratio(float(np.square(pp-yy).sum()), float(np.square(yy).sum()))
        return math.sqrt(value) if value is not None else None
    yc, vc = y-y.mean(), v-v.mean()
    corr_den = math.sqrt(float(np.square(yc).sum())*float(np.square(vc).sum()))
    return dict(nodes=n, hotspot_nodes=hotn, actual_signed_S_relative_L2=rel(v), actual_hotspot_relative_L2=rel(v, hot),
        magnitude_only_relative_L2=math.sqrt(magnitude/y2) if y2 > 0 else None,
        sign_oracle_relative_L2=rel(sign_oracle), magnitude_oracle_relative_L2=rel(magnitude_oracle),
        sign_oracle_hotspot_relative_L2=rel(sign_oracle, hot), magnitude_oracle_hotspot_relative_L2=rel(magnitude_oracle, hot),
        exact_sign_penalty_share_of_signed_SSE=ratio(penalty, signed),
        opposite_sign_error_share_of_signed_SSE=ratio(float(np.square(err[opposite]).sum()), signed),
        positive_truth_energy_share=ratio(float(np.square(y[y > 0]).sum()), y2),
        negative_truth_energy_share=ratio(float(np.square(y[y < 0]).sum()), y2),
        nearzero_truth_energy_share=ratio(float(np.square(y[near]).sum()), y2),
        nearzero_error_share=ratio(float(np.square(err[near]).sum()), signed),
        exactzero_truth_node_fraction=float(zero.mean()), exactzero_error_share=ratio(float(np.square(v[zero]).sum()), signed),
        opposite_sign_node_fraction=float(opposite.mean()), sign_disagreement_including_zero_fraction=float((np.sign(y) != np.sign(v)).mean()),
        predicted_to_true_field_norm_ratio=math.sqrt(p2/y2) if y2 > 0 else None,
        signed_cosine=ratio(float(np.dot(y, v)), math.sqrt(y2*p2)),
        centered_signed_correlation=ratio(float(np.dot(yc, vc)), corr_den),
        actual_signed_SSE_MPa2=signed, magnitude_only_SSE_MPa2=magnitude, exact_sign_penalty_MPa2=penalty,
        true_squared_norm_MPa2=y2, predicted_squared_norm_MPa2=p2,
        exact_decomposition_passed=True, oracle_is_deployable=False)


def summarize(rows, denominator):
    assert len(rows) == denominator
    fields = [k for k in next(iter(rows.values())) if k not in ('nodes', 'hotspot_nodes', 'exact_decomposition_passed', 'oracle_is_deployable')]
    return dict(fixed_denominator=denominator, per_case=rows,
        equal_case_means={f: sum(row[f] for row in rows.values())/denominator
                         if all(row[f] is not None and math.isfinite(row[f]) for row in rows.values()) else None for f in fields},
        complete_primary_metrics=all(row['actual_signed_S_relative_L2'] is not None for row in rows.values()))
