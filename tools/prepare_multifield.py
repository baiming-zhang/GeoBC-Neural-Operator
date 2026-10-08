#!/usr/bin/env python3
"""Add original signed uXYZ (mm) to preserved original DeepJEB prepared inputs.

The official H5 mirror can omit ver_x_disp(mm), and its nodal variables retain
CSV GRID ordering while its vertices are reordered. Coordinate matching aligns
all CSV source fields with H5/prepared vertices. Old prepared NPZs and existing
stress-only results are never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_original_csv(path: Path, load: str) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="") as source:
        header = next(csv.reader(source))
    names = (["nodeID"] + [f"coord_{axis}(mm)" for axis in "xyz"]
             + [f"{load}_{axis}_disp(mm)" for axis in "xyz"]
             + [f"{load}_resultant_disp(mm)", f"{load}_stress(MPa)"])
    columns = [header.index(name) for name in names]
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns, dtype=np.float64)
    return values, names


def prepare(raw: Path, existing: Path, output: Path, selection: list[str] | None = None,
            publish_manifest: bool = True) -> dict:
    if output.resolve() == existing.resolve():
        raise ValueError("Multifield output must differ from the preserved prepared-input directory.")
    old_manifest = json.loads((existing / "manifest.json").read_text())
    selection = selection or old_manifest["selection"]
    original_info = {item["sample_key"]: item for item in old_manifest["cases"]}
    output.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "DeepJEB original v1.0", "source_revision": "160424ca2a5095315ff9ead8a5fc36d2667fd3c6",
              "selection": selection, "feature_names": old_manifest["feature_names"],
              "feature_schema_note": old_manifest["feature_schema_note"],
              "normalization": old_manifest["normalization"], "cases": [],
              "shared_input_contract": {"xyz": "float32[N,3], normalized isotropic coordinates in [0,1]",
                                        "xyz_mm": "float32[N,3], original FEM node coordinates in mm",
                                        "features": "float32[N,25], existing exact deck/geometry/material schema",
                                        "u": "float32[N,3], original signed (ux,uy,uz) displacement components, mm",
                                        "stress": "float32[N], original source signed stress, MPa",
                                        "faces": "int32[M,3], global boundary connectivity from original TETRA10 nodes"},
              "target_policy": "Raw signed displacement components and raw source signed stress; no abs, no reconstructed VM, no interpolation/resampling of labels.",
              "source_h5_gap": "H5 vertices are reordered but nodal variables retain CSV GRID ordering; H5 may omit ver_x_disp(mm). Coordinate-align all raw CSV fields.",
              "preservation": "Existing prepared coordinates/features/faces copied exactly; stress corrected by CSV coordinate alignment in this NEW output; old inputs and runs untouched.",
              "label_alignment": "official CSV GRID coordinates -> H5 vertices"}
    for key in selection:
        cad, load = key.split("/")
        old_npz = existing / f"{cad}_{load}.npz"
        with np.load(old_npz, allow_pickle=False) as stored:
            data = {name: stored[name] for name in stored.files}
        csv_path = raw / "Field" / f"{cad}.csv"
        csv_values, csv_columns = load_original_csv(csv_path, load)
        xyz_mm, old_stress = data["xyz_mm"], data["stress"]
        if len(csv_values) != len(xyz_mm):
            raise ValueError(f"{key}: original CSV and prepared node counts differ")
        distances, indices = cKDTree(xyz_mm).query(csv_values[:, 1:4], workers=-1)
        if len(np.unique(indices)) != len(indices) or distances.max() > 2e-4:
            raise ValueError(f"{key}: CSV/prepared coordinate alignment is not one-to-one within tolerance")
        u = np.empty((len(xyz_mm), 3), dtype=np.float32)
        u[indices] = csv_values[:, 4:7].astype(np.float32)
        stress = np.empty(len(xyz_mm), dtype=np.float32)
        stress[indices] = csv_values[:, 8].astype(np.float32)
        old_stress_error = float(np.abs(stress.astype(np.float64) - old_stress).max())
        grid_node_ids = np.empty(len(xyz_mm), dtype=np.int64)
        grid_node_ids[indices] = csv_values[:, 0].astype(np.int64)
        if not np.isfinite(u).all() or data["features"].shape != (len(xyz_mm), 25):
            raise ValueError(f"{key}: invalid shared input/target shape")
        h5_path = raw / "FieldMesh" / f"{cad}.h5"
        component_checks, missing_h5 = {}, []
        with h5py.File(h5_path, "r") as field:
            raw_h5_stress = field[f"nodal_variables/{load}_stress(MPa)"][:]
            source_row_stress_error = float(np.abs(raw_h5_stress.astype(np.float64) - csv_values[:, 8]).max())
            if source_row_stress_error > 2e-4:
                raise ValueError(f"{key}: H5 source field does not match raw CSV GRID ordering within tolerance")
            for axis_index, axis in enumerate("xyz"):
                name = f"nodal_variables/{load}_{axis}_disp(mm)"
                if name not in field:
                    missing_h5.append(name)
                    continue
                error = float(np.abs(field[name][:].astype(np.float64) - csv_values[:, 4 + axis_index]).max())
                component_checks[axis] = error
                if error > 2e-6:
                    raise ValueError(f"{key}: original CSV/H5 displacement {axis} disagrees by {error} mm")
        data["u"] = u
        data["displacement"] = u
        data["stress"] = stress
        data["grid_node_ids"] = grid_node_ids
        data["label_alignment"] = np.asarray("official CSV GRID coordinates -> H5 vertices")
        data["u_units"] = np.asarray("mm")
        data["stress_units"] = np.asarray("MPa")
        target = output / old_npz.name
        np.savez(target, **data)
        info = {**original_info[key], "npz": str(target.resolve()), "preserved_input": str(old_npz.resolve()),
                "source_csv": str(csv_path.resolve()), "source_csv_sha256": sha256(csv_path),
                "u_component_source_columns": csv_columns[4:7], "u_shape": list(u.shape), "u_units": "mm",
                "u_min_mm": u.min(axis=0).tolist(), "u_max_mm": u.max(axis=0).tolist(),
                "u_RMS_mm": np.sqrt(np.mean(u.astype(np.float64)**2, axis=0)).tolist(),
                "csv_to_prepared_max_coordinate_error_mm": float(distances.max()),
                "old_prepared_wrong_alignment_stress_max_difference_MPa": old_stress_error,
                "h5_raw_field_vs_csv_GRID_row_stress_max_difference_MPa": source_row_stress_error,
                "CSV_to_H5_index_permutation_identity_fraction": float(np.mean(indices == np.arange(len(indices)))),
                "label_alignment": "official CSV GRID coordinates -> H5 vertices",
                "csv_vs_available_h5_displacement_max_difference_mm": component_checks,
                "missing_source_h5_displacement_fields": missing_h5}
        report["cases"].append(info)
        print(json.dumps({name: info[name] for name in ["sample_key", "u_shape", "u_min_mm", "u_max_mm", "missing_source_h5_displacement_fields", "csv_to_prepared_max_coordinate_error_mm"]}), flush=True)
    if publish_manifest:
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _build_one(arguments: tuple[str, str, str]) -> dict:
    """Build one isolated geometry input, then align CSV targets exactly once."""
    raw_value, output_value, key = arguments
    raw, output = Path(raw_value), Path(output_value)
    name = key.replace("/", "_")
    cached = output / ".case_manifests" / f"{name}.json"
    if cached.is_file() and (output / f"{name}.npz").is_file():
        report = json.loads(cached.read_text())
        if report.get("label_alignment") == "official CSV GRID coordinates -> H5 vertices":
            return report
    staging_root = (output / ".geometry_inputs").resolve()
    staging = (staging_root / name).resolve()
    if staging.parent != staging_root:
        raise ValueError("Unexpected geometry staging path")
    from tools.prepare_deepjeb import prepare as prepare_geometry
    with redirect_stdout(io.StringIO()):
        prepare_geometry(raw, staging, [key])
    report = prepare(raw, staging, output, [key], publish_manifest=False)
    report["cases"][0]["geometry_input_builder"] = "tools.prepare_deepjeb.prepare; isolated fresh deck-derived geometry/BC/material inputs"
    report["cases"][0].pop("preserved_input", None)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(report, indent=2) + "\n")
    # Only this worker's verified child directory is disposable.
    shutil.rmtree(staging)
    return report


def build_from_raw(raw: Path, output: Path, selection: list[str], workers: int = 3,
                   selection_policy: str | None = None) -> dict:
    """Prepare aligned fields from original raw components without old NPZ inputs."""
    if len(set(selection)) != len(selection):
        raise ValueError("Duplicate CAD/load selection")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(str(raw.resolve()), str(output.resolve()), key) for key in selection]
    if workers == 1:
        reports = [_build_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(_build_one, tasks))
    report = reports[0].copy()
    report["selection"] = selection
    report["cases"] = [entry for item in reports for entry in item["cases"]]
    report["preservation"] = "New raw-source geometry inputs and coordinate-aligned CSV fields; older prepared inputs/results untouched."
    report["selection_policy"] = selection_policy or "Explicit original DeepJEB CAD/load selection."
    report["nodes_total"] = sum(item["nodes"] for item in report["cases"])
    report["load_counts"] = {load: sum(key.endswith("/" + load) for key in selection)
                             for load in ["ver", "hor", "dia", "tor"]}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "nodes_total": report["nodes_total"],
                      "cases": len(selection), "load_counts": report["load_counts"]}), flush=True)
    return report


def archive_prefix_selection(raw: Path, count: int) -> list[str]:
    provenance = json.loads((raw / "deepjeb_provenance.json").read_text())
    record = provenance["archives"]["FieldMesh.tar.gz"]
    cad_ids = record["selected_cases"][:count]
    if len(cad_ids) != count:
        raise ValueError(f"Only {len(cad_ids)} of {count} requested archive-prefix H5 cases exist")
    return [f"{cad}/{['ver', 'hor', 'dia', 'tor'][index % 4]}"
            for index, cad in enumerate(cad_ids)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--existing", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared_multifield")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--build-inputs", action="store_true", help="Build isolated original geometry/BC inputs from raw H5/decks.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--archive-count", type=int, help="Choose the first N official archive-order CADs, round-robin ver/hor/dia/tor.")
    args = parser.parse_args()
    if args.build_inputs:
        selection = archive_prefix_selection(args.raw, args.archive_count) if args.archive_count else args.cases
        if not selection:
            parser.error("--build-inputs requires --cases or --archive-count")
        policy = ("Substitute first %s CADs in the pinned official HF archive order, round-robin ver/hor/dia/tor. The slides' seed-2026 subset of an unavailable 270-CAD pool is unknown and is not reconstructed."
                  % args.archive_count) if args.archive_count else None
        build_from_raw(args.raw, args.output, selection, args.workers, policy)
    else:
        prepare(args.raw, args.existing, args.output, args.cases)
