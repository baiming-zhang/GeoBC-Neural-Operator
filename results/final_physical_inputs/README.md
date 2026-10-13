# Final-set physical inputs

20/20 fixed final CADs have full-node physical displacement and gradient inputs; 60/60 endpoints were saved. New curved-P2 operators and degree-32 Chebyshev coarse corrections solve the three registered loads independently from zero. Recovered inputs are FP64 [Umm3, 1000*full-gradient9]. No final labels, model weights or raw source files are read in this stage.

Actual import/cache/operator/solve/gradient/save cost: UTC 613.901s / monotonic 562.007s. Source qualification is a separately reported cost. These are numerical input qualifications; model accuracy is measured only after all final predictions are saved.
