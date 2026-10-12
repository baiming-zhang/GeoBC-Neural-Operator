# Complete training physical inputs

Fixed-set input completion: 100/100 CADs, three canonical loads each. The new stage produced 12 independent zero-start fields and five full-node gradient inputs. It reused 95 previously qualified input arrays and three fields from the passed Chebyshev comparison.

New-stage cost: UTC 114.514s / monotonic 104.256s. The earlier budget-limited run and the paired comparison remain separate costs. The earlier 5000-second run was not reopened; the changed preconditioner was tested before producing the remaining fields.

Features are physical U(mm) and nine incident-volume-averaged P2 displacement-gradient components multiplied by 1000. Original-node Jacobians, incident coverage, finite values and independent physical-coordinate derivatives are checked. This establishes predictor availability, not original signed-S accuracy. Development-input qualification and a paired U/S learning experiment still follow.
