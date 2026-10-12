# Physical information comparison

The question is whether actual displacement and its full spatial gradient help the network learn the original signed U/S fields. Three fresh seeds each compare two models with identical parameters and initialization: ZERO replaces the 12 added input columns with zeros; ACTUAL supplies the qualified physical inputs. Both outputs remain learned.

Each arm uses the same shuffled case order for 20 passes over 100 training CADs and three loads. The architecture, Adam settings, physical relative loss and original training40 output scaler are shared. Runs save model, Adam, RNG and schedule position every 300 updates.

All six models' full-node predictions on the 20 known development CADs are saved before reading their shared original U/S targets. A useful improvement requires at least 3 percentage points in both mean primary and stress error, with primary wins in at least two seeds. These CADs have been iterated during research; this comparison is not a blind final test.

The physics input costs remain separate from neural training and inference. The original signed-stress definition is retained as a target; gradients are predictor information rather than a replacement stress formula.
