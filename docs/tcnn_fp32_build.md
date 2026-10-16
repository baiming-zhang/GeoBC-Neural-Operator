# Native TCNN FP32 build

The precision correction uses the existing tiny-cuda-nn source commit
`3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`, without downloads. It is built
as a separate runtime; the historical FP16 extension is retained.

Set `TCNN_HALF_PRECISION=0`, `TCNN_CUDA_ARCHITECTURES=120`, `MAX_JOBS=2`,
and the existing CUDA13.0 compiler/include/library paths. Build with
`python -m pip wheel --no-build-isolation --no-deps --no-index` against
`bindings/torch`, then install the wheel with `--no-index --no-deps --target`
into an isolated runtime. Prepend that runtime to `PYTHONPATH` for this run.
The source includes the existing C++20 compatibility adjustment used by the
historical build. No dependency versions or global installation were changed.

A first metadata attempt failed because CUDA_HOME/PATH were absent (54.79s).
After supplying the existing toolkit environment, build plus isolated install
took104.47s. These are build costs, separate from the training stage.

FP32 network precision uses the official CutlassMLP backend. The local
cutlass_matmul.h selects OpClassSimt with float compute/accumulators for
float network precision, rather than the half TensorOp specialization. Request it explicitly
in config; verify the native module dtype, actual native backend, registered and
active parameters, and full-node finite gradients. FP32 master parameters and
an output `.float()` cast alone do not establish native precision.

Set `NVIDIA_TF32_OVERRIDE=0` and disable PyTorch matmul/cuDNN TF32. This disables
TF32 acceleration; it does not promise bitwise reproducibility of every CUDA
reduction or strict IEEE behavior of all fast-math kernels. Native training
uses FP32 storage/computation without AMP. The old21 retained endpoints allow
TF32 and are labeled accordingly in the comparison.

Run the guarded trainer using configs/tcnn_fp32_u_overfitting.json and the
same TRAIN-only50 source mapping. All three arms start from seed2026 and
finish exactly5000 original-node updates. Only after native closure and
successful terminal/process checks, export once with:

```text
python -B -m tools.export_tcnn_fp32_u_overfitting --run runs/tcnn_fp32_u_overfitting
```

The exporter reads new run scalars and the previously exported1/5/50 CSVs.
It replaces only the three TCNN rows and retains their historical FP16 scalars
separately. Predictions, checkpoints, build binaries and source data stay local.
