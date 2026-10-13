# Final-set target-free sources

20/20 fixed final CADs have source-qualified geometry and three canonical load vectors. The reader uses H5 vertices/cells and only the original CSV GRID/XYZ columns, preserving the existing row permutation. Actual deck material, TET10 connectivity, zero constraints and RBE load transfer are checked. No U/S labels, physics solves, GPU work or model predictions are read in this stage.

Actual source/import/IO/save cost: UTC 143.078s / monotonic 131.256s. Source qualification precedes physical-input production; it is not an accuracy measurement.
