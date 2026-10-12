# Signed stress labels

The [DeepJEB paper](https://arxiv.org/html/2406.09047v2) identifies the node field as signed von Mises stress and describes an automated Inspire/SimLab pipeline.

Current [SimLab computed-results documentation](https://help.altair.com/simlab/help/en_us/topics/results/computed_results.htm) assigns the sign of the principal stress with largest absolute magnitude. Its [averaging documentation](https://help.altair.com/simlab/help/en_us/topics/results/userinterface_results_averagingmethod.htm) distinguishes computing invariants before averaging from averaging corner/mid-node quantities before computing invariants.

These product definitions narrow the plausible post-processing path. The original DeepJEB export session, selected averaging method and software-version settings are still missing, so they do not establish a replacement formula for the source CSV. Training keeps the published signed scalar unchanged.

The physical-input production limit concerns solve cost, independently of this label-definition gap. A bounded coarse-preconditioner comparison is the next numerical experiment.
