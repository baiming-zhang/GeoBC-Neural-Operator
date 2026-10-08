# Reference and reproduction scope

Reference: https://docs.google.com/presentation/d/1g5S0QCZSDR_igYFWFe20SQ3nw2yvgYbXk_t4QSmKpBU/edit

The reference was read through the authenticated Google Drive connector on 2026-10-08. Its original text and speaker notes are saved in `slides_text.json`; the full resource is saved locally in `slides_source.json`.

The current scope is all recoverable quantitative slide experiments, including MLP, NGP, Transolver/++/3, output residuals, 64³/192³ FNO, distillation, and shared 50-case hash fields. One-case and five-case fitting remain the first priority. The exact 30-job protocol is recorded in `../notes/all_methods_plan.json`. Five cases and original Transolver are supplementary controls.

Reference configuration: 12 hash levels, 2 features per level, log2 hash table size 19, base resolution 8, per-level growth 1.6, a width-64 MLP with two hidden layers, normalized coordinates in [0,1], invertible asinh stress scaling, 1,000 sampled updates. Shared fitting uses 25 extra geometry/BC/material features without a sample-ID. The original implementation and the exact 25-feature schema/scaling constants are not included in the slides.

Reference fitting errors (decoded physical stress relative L2): single case `506_533/ver` 0.00077296799; two cases vertical/torsion approximately 0.00088/0.00107686. The one-case reference contains 120,777 original FEM nodes and selects its best checkpoint at update 600. The two-case reference balances 16,384 nodes/case/update and evaluates all 243,564 nodes.

The data must be the original DeepJEB v1.0, not DeepJEB++. The target is the source signed stress scalar in MPa, not von Mises stress and not a tensor reconstructed from a scalar. No grid resampling or teacher is used in these initial fits. Full-node metrics are reported separately from hotspot metrics; fitting results do not establish unseen-geometry or remeshing generalization.

The official Drive source supplied the exact slide CADs `506_533` and `506_25`: 243,564 body FEM nodes in total. Original H5, Field CSV and solver decks are pinned by file IDs, sizes and SHA256 in `deepjeb_exact_files.json`. CSV GRID coordinates align the source uXYZ and signed stress to H5 vertices; the H5-only label order is not valid and the earlier physical-field claims were withdrawn.

The five-case suite adds `10_30/ver`, `10_421/hor`, and `10_460/dia`, totaling 791,820 nodes. The 50-case suite uses the first 50 CADs of the pinned official archive, round-robin load types, totaling 10,684,820 nodes. This is an explicit substitute for the unavailable historical seed-2026 subset of a 270-CAD training pool; it cannot be called the same split.

The original run directory is unavailable. Unpublished scalers, optimizer settings, TS3 dimensions, feature definitions and grid origin are explicitly recorded as gaps or replacement assumptions. No exact numerical reproduction is claimed where those details remain unknown. Only training-case fitting is tested.
