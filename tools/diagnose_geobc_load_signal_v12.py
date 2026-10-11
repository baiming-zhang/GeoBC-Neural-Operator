"""Input-only inspection: no target arrays, inference, Torch or CUDA."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def run():
    import numpy as np
    began = time.perf_counter()
    path = Path('data/geobc_prospective_dataset_v1/manifest.json')
    manifest = json.loads(path.read_text())
    records = manifest['records']; train = manifest['train_keys']['100']
    cads = sorted({records[key]['cad_id'] for key in train})
    assert len(cads) == 100
    ordered = sorted(cads, key=lambda cad: (records[cad+'/ver']['num_nodes'], cad))
    cad = ordered[(len(ordered)-1)//2]
    keys = [cad+'/'+load for load in ('ver', 'hor', 'tor')]
    assert all(key in train for key in keys)
    rows = []
    for key in keys:
        record = records[key]; filename = Path(record['features_path'])
        identity = record['file_identities']['features_path']; stat = filename.stat()
        assert stat.st_size == identity['bytes'] and stat.st_mtime_ns == identity['mtime_ns']
        array = np.load(filename, mmap_mode='r', allow_pickle=False)
        assert array.shape == (record['num_nodes'], 25)
        active = array[:, 6] > 0
        assert active.any()
        vectors = np.unique(array[active, 7:13], axis=0)
        rows.append(dict(key=key, nodes=len(array), loaded_nodes=int(active.sum()),
            loaded_fraction=float(active.mean()),
            outside_load_vector_nonzero=int(np.count_nonzero(array[~active, 7:13])),
            active_normalized_load_vectors=vectors.tolist(),
            constant_load_vector_on_active_nodes=bool(len(vectors)==1), features_identity=identity))
        del array
    receipt = dict(status='actual_input_only_load_signal_diagnostic',
        observed_UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='TRAIN100 lower median node count; CAD ID tie break; metadata only selection',
        selected_CAD=cad, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows,
        CPU_seconds=time.perf_counter()-began, numeric_GT_reads=0, PT_reads=0,
        GPU_calls=0, model_inference_calls=0,
        interpretation='Inspect outside-load nonzero counts before inferring localization; no accuracy gain measured',
        previous_inline_metadata_failure='train_keys is a dict of 40/100 lists; failed before any array read',
        no_new_GPU_execution_released=True)
    output = Path('notes/geobc_global_load_signal_input_diagnostic_v12.json')
    assert not output.exists()
    output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','manifest_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--inspect-inputs', action='store_true')
    args = parser.parse_args()
    if args.inspect_inputs:
        run()
