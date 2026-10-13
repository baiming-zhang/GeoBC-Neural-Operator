# Query prepared U/S responses

The compact response tool accepts physical XYZ coordinates in millimetres and one of the registered `ver`, `hor` or `tor` loads. It uses the first registered trained seed (2026), without selecting a seed using final errors.

Prepare a private catalog containing `seed: 2026`, `input_mode: FIELDWISE`, the model `checkpoint`, the original training40 `scaler`, the scientific split role, and `cases` keyed by CAD ID. Each case supplies its FP64 `vertices_mm`/int64 `cells10` geometry archive, `nodes`, `cells`, registered `reference_load6`, the physical-input descriptor and the three target-free predictor records. The physical covariates are `[Umm3, 1000*gradient9]`; their preparation costs are reported separately. The local final20 catalog is generated from the completed input indexes. Catalogs, data and weights stay private.

The metadata-only catalog builder reads completed indexes and source facts, without model or scientific array bodies:

```sh
python -B -m tools.prepare_response_catalog \
  --predictors /path/to/final_model_sources.json \
  --geometry-index runs/final_targetfree_sources/source_index.json \
  --physical-index runs/final_physical_inputs/complete_inputs.json \
  --output /path/to/new_prepared_response_catalog.json
```

Write one JSON request per line:

```json
{"cad":"21_119","load":"ver","xyz_mm":[[12.0,34.0,56.0]]}
```

The coordinates above illustrate the format; they are not asserted to lie inside that CAD. Each request may contain 1 to 4096 points. A rejected domain request returns null fields for the entire batch; unsupported CADs, loads, malformed or nonfinite coordinates return `invalid_request` before loading a CAD. No element IDs are supplied by the caller.

Run from the repository with CUDA and the prepared private catalog:

```sh
python -B -m tools.query_response \
  --catalog /path/to/prepared_response_catalog.json \
  --requests /path/to/requests.jsonl \
  --output /path/to/new_responses.jsonl \
  --lock /path/to/geobc.coordinator.lock
```

The output file must be new. A successful response contains `status: completed`, `U_mm` as three components per point, and `signed_S_MPa` as one signed scalar per point. No stress-sign convention is invented. A single model is loaded for the session; the current CAD's inputs and three load contexts are reused. Group requests by CAD to avoid repeated CAD loading. The default whole-session allowance is 900 seconds with a 30.5 GB device-memory bound.

Final20 mean errors across all three registered seeds are U 11.629% and S 34.247%. The first-seed final errors are U 11.565% and S 34.264%. These aggregate nodal errors do not provide an error bound for an individual query. Continuous input interpolation has no independent off-node truth qualification, C1 guarantee or arbitrary-load/changed-BC support.
