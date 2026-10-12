# Data

Original DeepJEB CSV GRID coordinates align with all original H5 vertices. U is in mm and the original signed scalar S is in MPa. Keep the output scaler fitted on the original 40 training CADs.

The working protocol uses 100 training CADs, 20 known development CADs and 20 already revealed final CADs, each with vertical, horizontal and torsional loads. CAD IDs are separated; shape-family separation has not been established. Development results are iterated, so they are not a blind test.

Raw data, generated arrays and trained checkpoints stay in the local data/ and runs/ directories. Prepared physics caches contain vertices_mm, cells10, GRID mapping, free/fixed nodes, RBE transfer and physical loads. Target labels are read explicitly for supervised experiments.
