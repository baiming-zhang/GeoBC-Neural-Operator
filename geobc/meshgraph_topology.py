"""Original quadratic mesh edges; no target fields are needed to build this graph."""
import numpy as np

TETRA10_EDGES=((0,1,4),(1,2,5),(2,0,6),(0,3,7),(1,3,8),(2,3,9))

def quadratic_mesh_edges(cells,node_count):
 cells=np.asarray(cells)
 if cells.ndim!=2 or cells.shape[1]!=10 or not len(cells) or cells.dtype.kind not in 'iu' or cells.min()<0 or cells.max()>=node_count:raise ValueError('Original zero-based TETRA10 cells required')
 pieces=[]
 for a,b,m in TETRA10_EDGES:pieces.extend((cells[:,[a,m]],cells[:,[m,b]]))
 pairs=np.concatenate(pieces).astype(np.int64);pairs.sort(axis=1)
 if np.any(pairs[:,0]==pairs[:,1]):raise ValueError('Degenerate original mesh edge')
 codes=np.unique(pairs[:,0]*node_count+pairs[:,1]);u=codes//node_count;v=codes%node_count
 send=np.concatenate((u,v));recv=np.concatenate((v,u));order=np.lexsort((send,recv));edge=np.stack((send[order],recv[order]),axis=1)
 degree=np.bincount(edge[:,1],minlength=node_count)
 if np.any(degree==0):raise ValueError('Original nodes must not be omitted or isolated')
 return edge.astype(np.int32),dict(nodes=int(node_count),tetra10_cells=len(cells),undirected_edges=len(codes),directed_edges=len(edge),minimum_degree=int(degree.min()),maximum_degree=int(degree.max()),mean_degree=float(degree.mean()),isolated_nodes=0)
