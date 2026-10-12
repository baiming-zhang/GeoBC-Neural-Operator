# Fieldwise continuous U/S response

Interface checks passed on the fixed three CADs. 432 physical U/S query rows were saved, including 360 interior rows and 72 nodal witnesses. 9 outside-box requests were explicitly rejected.

The API takes physical coordinates in mm and a registered canonical load. It locates the curved element, interpolates native condition features and the recovered physical covariates, and queries the trained FNO decoder. U comes from the U_ONLY model and S from the GRAD_ONLY model, sharing point location and input interpolation. Two networks contain 9,056,788 parameters in total. The routes were selected from development results; the first seed is used. Coordinate gradients include the actual inverse and covariate interpolation. Checks compare the located API with the same model queried in explicit cells and with saved nodal predictions; no new off-node FEM accuracy is measured.

Actual cache/model/index/query/check/save cost: UTC 61.252s / monotonic 55.337s. Physics preprocessing and neural training costs are inherited separately.
