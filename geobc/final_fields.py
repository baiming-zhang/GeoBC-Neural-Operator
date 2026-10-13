"""Read original signed final fields only after the prediction barrier."""
import csv
import numpy as np

LOADS=['ver','hor','tor']


def read_final_fields(path,permutation,expected_grid,prediction_barrier):
    if not prediction_barrier:
        raise RuntimeError('All selected final predictions must be saved first')
    with path.open(encoding='utf-8-sig',newline='') as file:
        header=next(csv.reader(file))
    names=['nodeID']+[load+suffix for load in LOADS for suffix in ['_x_disp(mm)','_y_disp(mm)','_z_disp(mm)','_stress(MPa)']]
    if len(header)!=len(set(header)) or any(name not in header for name in names):
        raise ValueError('Original signed field columns differ')
    data=np.loadtxt(path,delimiter=',',skiprows=1,usecols=[header.index(name) for name in names],dtype=np.float64,ndmin=2)
    n=len(expected_grid)
    if data.shape!=(n,13) or not np.isfinite(data).all() or permutation.dtype!=np.int64 or permutation.shape!=(n,) or not np.array_equal(np.sort(permutation),np.arange(n)):
        raise ValueError('Final fields or original row mapping differ')
    if not np.array_equal(data[permutation,0],expected_grid):
        raise ValueError('Final CSV GRID rows differ from qualified geometry')
    values=np.ascontiguousarray(data[permutation,1:],dtype=np.float32)
    if not np.isfinite(values).all():raise FloatingPointError('Final field FP32 representation overflows')
    return {load:values[:,4*i:4*i+4] for i,load in enumerate(LOADS)}
