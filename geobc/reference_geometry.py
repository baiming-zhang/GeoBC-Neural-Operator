import numpy as np
EDGES=np.array(((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)),dtype=np.int64)
_DL=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

def _bary(values):
    result=np.asarray(values)
    if result.shape[-1:]!=(4,) or not np.isfinite(result).all():
        raise ValueError('Expected finite barycentric coordinates [...,4]')
    return result if result.dtype.kind=='f' else result.astype(np.float64)

def shape_gradient(barycentric):
    """dN/dr for r=(L1,L2,L3), shape [...,10,3]."""
    b = _bary(barycentric)
    d = _DL.astype(b.dtype)
    corners = (4 * b - 1)[..., :, None] * d
    mids = 4 * (b[..., EDGES[:, 1], None] * d[EDGES[:, 0]] + b[..., EDGES[:, 0], None] * d[EDGES[:, 1]])
    return np.concatenate((corners, mids), axis=-2)
