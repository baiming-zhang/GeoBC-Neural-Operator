"""Summarize existing acquisition/preparation manifests without rereading fields."""
from datetime import datetime
import json
from pathlib import Path


def main():
    provenance = json.loads(Path("data/raw/deepjeb_provenance.json").read_text())
    archives = {}
    for filename in ["FieldMesh.tar.gz", "Field.tar.gz", "SolverDeck.tar.gz"]:
        entry = provenance["archives"][filename]
        archives[filename] = {"selected_CADs": len(entry["selected_cases"]),
                              "compressed_prefix_bytes": entry["cached_compressed_bytes"],
                              "new_network_bytes": entry["downloaded_bytes_this_run"],
                              "elapsed_seconds": (datetime.fromisoformat(entry["finished_at"])
                                                  - datetime.fromisoformat(entry["started_at"])).total_seconds()}
    directories = {}
    for directory in ["prepared_exact_multifield", "prepared_benchmark5", "prepared50_multifield"]:
        path = Path("data") / directory
        manifest = json.loads((path / "manifest.json").read_text())
        nodes = sum(case["nodes"] for case in manifest["cases"])
        directories[str(path)] = {
            "selection": manifest["selection"], "CADs": len(manifest["cases"]), "nodes": nodes,
            "conditioning_channels": len(manifest["feature_names"]),
            "max_CSV_to_H5_coordinate_matching_error_mm": max(case["csv_to_prepared_max_coordinate_error_mm"] for case in manifest["cases"]),
            "max_H5_raw_fields_vs_CSV_GRID_row_stress_error_MPa": max(case["h5_raw_field_vs_csv_GRID_row_stress_max_difference_MPa"] for case in manifest["cases"]),
            "label_alignment": manifest["label_alignment"],
            "selection_policy": manifest.get("selection_policy"),
            "steps_per_true_global_epoch_at_51200": (nodes + 51199) // 51200,
            "equivalent_full_passes_at_10000_updates_of_51200": 10000 * 51200 / nodes,
        }
    official_splits = {path.name: len(json.loads(path.read_text()))
                       for path in Path("data/raw/Metadata").glob("*.json")}
    report = {"dataset": "Original DeepJEB v1.0; no DeepJEB++ files",
              "official_source_page": "https://dataset.narnia.ai/deepjeb.html",
              "official_Drive_root": "https://drive.google.com/drive/folders/10ccsas7TfD7nIan-Ll5y9vbx4-tSqEyJ",
              "pinned_HF_revision": provenance["revision"],
              "archives": archives, "prepared_directories": directories,
              "official_metadata_split_counts": official_splits,
              "unavailable_reference_split": "Slides use a bespoke 300-CAD subset with 270 train / 30 held out. The official release metadata has the counts above, not the slides' subset. The exact seed-2026 fifty selected from its unknown 270-CAD pool cannot be recovered.",
              "field_policy": "All targets are raw signed CSV uXYZ(mm) and source signed stress(MPa), uniquely coordinate-aligned to reordered H5 vertices. Old same-index H5-field/vertex pairs are invalid and preserved separately.",
              "feature_schema_gap": "Slides' historical 25-feature preprocessing implementation is unavailable. Current 25 features are explicitly reconstructed from official decks, material and geometry with no labels.",
              "surface_policy": "Original TETRA10 boundary with quadratic midside nodes retained; H5 surface-local face indexing is not applied directly to volume vertices.",
              "exact_case_provenance": "data/raw_exact/deepjeb_drive_provenance.json",
              "all_public_Drive_filename_index": "notes/original_deepjeb_drive_index.json"}
    Path("notes/dataset_readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prepared": {key: {name: val[name] for name in ["CADs", "nodes", "conditioning_channels", "steps_per_true_global_epoch_at_51200"]} for key, val in directories.items()}, "downloaded_compressed_prefix_bytes": sum(value["compressed_prefix_bytes"] for value in archives.values())}, indent=2))


if __name__ == "__main__":
    main()
