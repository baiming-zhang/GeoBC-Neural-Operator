"""Local cached-wheel recovery; no network or repeated dependency acquisition."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run():
    began=time.time();out=ROOT/'notes/geobc_xgboost_cached_recovery_v92_receipt.json'
    assert sys.executable=='/home/research/.venvs/geobc/bin/python' and not out.exists()
    old=json.loads((ROOT/'notes/geobc_xgboost_dependency_v92_receipt.json').read_text())
    inspection=json.loads((ROOT/'notes/geobc_xgboost_dependency_v92_cached_wheel_inspection.json').read_text())
    assert old['status']=='failed' and old['error_type']=='TimeoutExpired'
    wheel=ROOT/inspection['wheel']['path'];assert hashlib.sha256(wheel.read_bytes()).hexdigest()==inspection['wheel']['sha256']
    state=dict(status='running',started_at_epoch=began,prior_failed_install_seconds=old['elapsed_epoch_seconds'],
        recovery_budget_seconds=60,network_requests=0,scientific_arrays_GT_PT_GPU=0,actual_GPU_training_verified=False,
        scope='No-deps offline cached binary install; existing numpy/scipy kept; actual dynamic dependency inspection found no requiredNCCL. SingleGPU runtime still must be verified in real experiment.')
    def save():
        tmp=out.with_name(out.name+'.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');os.replace(tmp,out)
    save()
    try:
        assert importlib.metadata.version('numpy') and importlib.metadata.version('scipy')
        try: current=importlib.metadata.version('xgboost')
        except importlib.metadata.PackageNotFoundError:current=None
        assert current is None or current=='3.0.5'
        folder=ROOT/'runs/dependency_cache_v92/local_wheels';folder.mkdir(exist_ok=True)
        named=folder/'xgboost-3.0.5-py3-none-manylinux_2_28_x86_64.whl'
        if not named.exists():os.link(wheel,named)
        assert hashlib.sha256(named.read_bytes()).hexdigest()==inspection['wheel']['sha256']
        env={k:v for k,v in os.environ.items() if not k.startswith('PIP_')};env['PIP_CONFIG_FILE']='/dev/null';env['PYTHONNOUSERSITE']='1'
        with (ROOT/'notes/geobc_xgboost_cached_recovery_v92_install.log').open('x') as log:
            code=subprocess.run([sys.executable,'-m','pip','install',str(named),'--no-index','--no-deps',
                '--only-binary=:all:','--disable-pip-version-check','--no-input'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=max(1,60-(time.time()-began))).returncode
        state['pip_exit_code']=code;assert code==0
        import xgboost
        assert xgboost.__version__=='3.0.5' and xgboost.build_info()['USE_CUDA']
        lib=Path(xgboost.core._LIB._name)
        state.update(status='completed',installed_version='3.0.5',build_info=xgboost.build_info(),
            installed_library_path=str(lib),installed_library_bytes=lib.stat().st_size,
            installed_library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
            compiled_GPU_support=True,actual_GPU_training_verified=False,dependency_namespace_import_verified=True,
            cached_wheel_sha256=inspection['wheel']['sha256'])
    except BaseException as err:
        state.update(status='failed',error_type=type(err).__name__,error=str(err),automatic_retry=False);raise
    finally:
        state['elapsed_epoch_seconds']=time.time()-began;state['cumulative_dependency_setup_seconds']=old['elapsed_epoch_seconds']+state['elapsed_epoch_seconds'];save()
    print(json.dumps({k:state[k] for k in ('status','installed_version','elapsed_epoch_seconds','cumulative_dependency_setup_seconds','network_requests','actual_GPU_training_verified')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--install-offline',action='store_true')
    if p.parse_args().install_offline:run()
    else:print('{"status":"source_only","network_GPU_GT":0}')
