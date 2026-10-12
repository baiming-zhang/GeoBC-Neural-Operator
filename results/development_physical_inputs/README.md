# Development physical inputs

All 20 known development CADs and 60 canonical fields now have physical U and full-gradient covariates. The largest geometry has 352,321 original nodes. Maximum independent physical-coordinate derivative difference: 2.21e-14.

Actual cost: UTC 62.402s / monotonic 56.491s. Existing converged U and positive curved-cell volumes were loaded once; no elasticity solve or ground-truth read was needed. Features use three U(mm) components and nine volume-averaged P2 gradients multiplied by 1000. Source and solve costs belong to the earlier displacement results. This is input qualification; signed-stress learning remains to be tested.
