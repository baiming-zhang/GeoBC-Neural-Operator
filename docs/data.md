# Data

Original DeepJEB CSV GRID coordinates align with all original H5 vertices. U is in mm and the original signed scalar S is in MPa. Keep the output scaler fitted on the original 40 training CADs.

The protocol uses 100 training CADs, 20 iterated development CADs and 20 fixed final CADs, each with vertical, horizontal and torsional loads. Six selected model endpoints (ZERO and FIELDWISE, three seeds each) saved all 360 full-node final predictions before the pipeline first parsed the final numerical U/S fields, once per CAD for all three loads. Final labels are now revealed and cannot be used for model or threshold tuning. CAD IDs are separated; shape-family separation has not been established. This availability-conditional split lies within the official training pool and does not establish global blind evaluation.

Raw data, generated arrays and trained checkpoints stay in the local data/ and runs/ directories. Prepared physics caches contain vertices_mm, cells10, GRID mapping, free/fixed nodes, RBE transfer and physical loads. Target labels are read explicitly for supervised experiments.
