# Continuous learned U/S response

Interface checks passed on the fixed three CADs. 360 physical U/S query rows were saved, including 288 interior rows and 72 nodal witnesses. 9 outside-box requests were explicitly rejected.

The API takes physical coordinates in mm and a registered canonical load. It locates the curved element, interpolates native condition features and the recovered physical covariates, and queries the trained FNO decoder. The first preregistered seed is used. Coordinate gradients include the actual inverse and covariate interpolation. Checks compare the located API with the same model queried in explicit cells and with saved nodal predictions; no new off-node FEM accuracy is measured.

Actual cache/model/index/query/check/save cost: UTC 26.750s / monotonic 23.795s. Physics preprocessing and neural training costs are inherited separately.
