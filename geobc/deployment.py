"""Prepared-cache XYZ response session, with no target-field reader."""
import gc
import json
import numpy as np
import torch
from .fieldwise_neural_field import FieldwiseCovariateFNO
from .learning_data import LearningCache
from .neural_response import LocatedNeuralResponse
from .physics_inputs import local_path


def validate_request(request, registered_CADs):
    if not isinstance(request,dict) or set(request)!={'cad','load','xyz_mm'}:
        raise ValueError('Request requires cad, load and xyz_mm')
    if not isinstance(request['cad'],str) or request['cad'] not in registered_CADs:
        raise ValueError('CAD is not registered')
    if request['load'] not in ['ver','hor','tor']:
        raise ValueError('Load is not registered')
    rows=request['xyz_mm']
    if not isinstance(rows,list) or not 1<=len(rows)<=4096 or any(not isinstance(row,list) or len(row)!=3 or any(type(x) not in (int,float) for x in row) for row in rows):
        raise ValueError('xyz_mm requires 1 to 4096 numeric XYZ rows')
    xyz=np.asarray(rows,dtype=np.float64)
    if not np.isfinite(xyz).all():raise ValueError('XYZ coordinates must be finite')
    return xyz


class PreparedResponseSession:
    """One fixed model, with one CAD and its three canonical contexts resident."""
    def __init__(self,catalog,guard):
        self.catalog,self.guard=catalog,guard
        self.counts=dict(checkpoint_reads=0,geometry_reads=0,feature_reads=0,predictor_reads=0,GT_reads=0,training_updates=0,scaler_fits=0)
        self.current_CAD=None;self.api=None;self.cache=None
        if catalog['seed']!=2026 or catalog['input_mode']!='FIELDWISE':
            raise ValueError('First registered compact-model endpoint required')
        guard();checkpoint=torch.load(local_path(catalog['checkpoint']),map_location='cpu',weights_only=False)
        self.counts['checkpoint_reads']+=1
        if checkpoint['arm']!='fieldwise_seed2026' or checkpoint['cursor']!=6000:
            raise ValueError('Registered model endpoint differs')
        self.model=FieldwiseCovariateFNO(2026).cuda();self.model.load_state_dict(checkpoint['model']);self.model.eval()
        del checkpoint;guard()

    def open_CAD(self,cad):
        if cad not in self.catalog['cases']:raise ValueError('CAD is not registered')
        if cad==self.current_CAD:return self.api
        self.api=None;self.cache=None;self.current_CAD=None
        gc.collect();torch.cuda.empty_cache();self.guard()
        record=self.catalog['cases'][cad];n,c=record['nodes'],record['cells']
        with np.load(local_path(record['geometry']),allow_pickle=False) as source:
            vertices,cells=source['vertices_mm'],source['cells10']
            if vertices.dtype!=np.float64 or vertices.shape!=(n,3) or cells.dtype!=np.int64 or cells.shape!=(c,10) or not np.isfinite(vertices).all() or cells.min()<0 or cells.max()>=n:
                raise ValueError('Prepared geometry schema differs')
            vertices=torch.as_tensor(vertices,device='cuda');cells=torch.as_tensor(cells,device='cuda')
        self.counts['geometry_reads']+=1
        cache=LearningCache({},self.guard)
        cache.load_predictors(record['predictors'],{cad:record['physical_inputs']},self.catalog['scientific_group'],keep_physical_covariates=True)
        self.counts['feature_reads']+=cache.feature_reads;self.counts['predictor_reads']+=cache.input_reads
        cases={load:cache.cases[cad+'/'+load] for load in ['ver','hor','tor']}
        B=torch.tensor([record['reference_load6'][load] for load in ['ver','hor','tor']],device='cuda',dtype=torch.float64).T
        self.api=LocatedNeuralResponse(self.model,vertices,cells,cache.physical_covariates[cad],B,cases,self.catalog['scaler'],self.guard)
        self.cache=cache;self.current_CAD=cad;self.guard()
        return self.api

    def query(self,request):
        xyz=validate_request(request,self.catalog['cases'])
        api=self.open_CAD(request['cad'])
        with torch.no_grad():result=api.query(torch.as_tensor(xyz,device='cuda'),request['load'])
        self.guard()
        response=dict(cad=request['cad'],load=request['load'],status=result['status'],U_mm=None,signed_S_MPa=None)
        if result['physical'] is not None:
            physical=result['physical'].cpu().numpy()
            response.update(U_mm=physical[:,:3].tolist(),signed_S_MPa=physical[:,3].tolist())
        else:response['reason']='At least one point is outside the supported domain or has ambiguous location'
        return response


def handle_line(session,line):
    """Parse one public JSON request; unsupported inputs return explicit null fields."""
    try:
        request=json.loads(line)
        validate_request(request,session.catalog['cases'])
    except (ValueError,TypeError,OverflowError) as error:
        return dict(status='invalid_request',U_mm=None,signed_S_MPa=None,reason=str(error))
    return session.query(request)
