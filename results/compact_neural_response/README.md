# Compact continuous U/S response

Interface checks passed on the fixed three CADs. 504 physical U/S query rows were saved, including 432 interior rows and 72 nodal witnesses. 9 outside-box requests were explicitly rejected.

The API takes physical coordinates in mm and a registered canonical load. It locates the curved element, interpolates native condition features and the recovered physical covariates, and queries the trained FNO decoder. A single 4,554,762-parameter model shares its geometry/FNO trunk, with U3 inputs for the U head and gradient9 inputs for the S head. The first registered seed is used. Coordinate gradients include the actual inverse and covariate interpolation. Checks compare the located API with the same model queried in explicit cells and with saved nodal predictions; no new off-node FEM accuracy is measured.

Actual cache/model/index/query/check/save cost: UTC 31.852s / monotonic 28.896s. Physics preprocessing and neural training costs are inherited separately.
