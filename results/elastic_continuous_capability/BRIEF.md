# Physical native TET10 continuous U / gradient

Every original cell/fourQ uses FP64 mm geometry and original zero-start U.
The reference is the same P2 evaluation of GT nodal U; strain is a derived proxy.

| CAD/ver | FullQ U L2 | Strain Frobenius proxy L2 | Cached query s |
|---|---:|---:|---:|
| 101_428/ver | 0.0162% | 0.1989% | 1.861 |
| 101_56/ver | 0.0162% | 0.1935% | 1.397 |
| 101_78/ver | 0.0162% | 0.1596% | 1.067 |
| 102_249/ver | 0.0162% | 0.1178% | 1.125 |
| 102_38/ver | 0.0162% | 0.1456% | 1.047 |

Coverage/integrity gate: True; fixed denominator5.
Normal derivative jumps are expected C0 finite-element diagnostics; C1 is not required.
Native all-cell barycentric evaluation and a bounded32-cell inverse were checked; a fast whole-CAD locator is not claimed.
New actual wall 14.231s; original GT-free physical452.954s inherited once.
Deployment cold costs compose original physics and measured query costs; source GT metrics are separate.
