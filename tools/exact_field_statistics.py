"""Read-only target statistics for slide 7 using the actual benchmark scaler."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from geobc.benchmark import FieldScaler


def main():
    source = Path("data/prepared_exact_multifield")
    keys = ["506_533/ver", "506_25/tor"]
    targets = []
    for key in keys:
        with np.load(source / (key.replace("/", "_") + ".npz"), allow_pickle=False) as data:
            if data["label_alignment"].item() != "official CSV GRID coordinates -> H5 vertices":
                raise ValueError("Unaligned target archive")
            targets.append(np.column_stack((data["displacement"], data["stress"])))
    scaler = FieldScaler.fit(targets, [0, 1, 2, 3], "linear")
    encoded_cases = [scaler.encode(target).astype(np.float64) for target in targets]
    def statistics(encoded, physical):
        stress = encoded[:, 3]
        top_count = int(np.ceil(len(stress) * .01))
        selected = np.argpartition(np.abs(stress), len(stress) - top_count)[-top_count:]
        physical_selected = np.argpartition(np.abs(physical[:, 3]), len(stress) - top_count)[-top_count:]
        energy = np.sum(encoded ** 2, axis=0)
        return {"nodes": len(encoded), "top_1pct_node_count": top_count,
                "max_abs_standardized_stress": float(np.abs(stress).max()),
                "top_1pct_stress_nodes_target_energy_share": float(np.sum(stress[selected] ** 2) / np.sum(stress ** 2)),
                "top_1pct_by_abs_physical_stress_standardized_energy_share": float(np.sum(stress[physical_selected] ** 2) / np.sum(stress ** 2)),
                "stress_fraction_four_standardized_channel_target_energy": float(energy[3] / energy.sum()),
                "channel_target_energy": energy.tolist()}
    encoded = np.concatenate(encoded_cases)
    values = statistics(encoded, np.concatenate(targets))
    per_case = {key: statistics(value, target) for key, value, target in zip(keys, encoded_cases, targets)}
    reference = {"max_abs_standardized_stress": 16.4053,
                 "top_1pct_stress_nodes_target_energy_share": .239874,
                 "stress_fraction_four_standardized_channel_target_energy": .267}
    vertical = per_case["506_533/ver"]
    report = {"cases": keys, "nodes": len(encoded),
              "scaler_implementation": "geobc.benchmark.FieldScaler.fit(targets,[0,1,2,3],'linear')",
              "scaler": asdict(scaler), "pooled": values, "per_case": per_case,
              "slide_7_reference": reference, "reference_measurement_case": "506_533/ver",
              "vertical_minus_reference": {name: vertical[name]-reference[name] for name in reference},
              "definition": "Fit per-channel mean/population-std on pooled two-case training fields, then measure pooled and each individual case separately. Top ceil(1% N) by |standardized source signed stress|; also report physical-stress ranking as a separately labeled definition. Target energy, not prediction residual energy.",
              "interpretation": "Slide 7 measures the vertical case under the pooled training scaler. The pooled four-channel share is 25% by normalization; an individual-case share need not be 25%. No scaler is adjusted to match the slide values."}
    Path("notes/exact_processed_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
