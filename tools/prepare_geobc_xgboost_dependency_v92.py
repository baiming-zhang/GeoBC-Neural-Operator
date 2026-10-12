"""One bounded binary dependency install in the named research virtualenv."""
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
RECEIPT = ROOT/'notes/geobc_xgboost_dependency_v92_receipt.json'
VERSION = '3.0.5'


def prepare():
    began = time.time()
    assert sys.executable == '/home/research/.venvs/geobc/bin/python'
    assert not RECEIPT.exists(), 'Do not replay dependency attempt'
    state = dict(status='running',started_at_epoch=began,version=VERSION,maximum_seconds=180,
        scientific_arrays_GT_PT_GPU=0,actual_training_GPU_verified=False)

    def save():
        temp = RECEIPT.with_name(RECEIPT.name+'.tmp')
        temp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,RECEIPT)

    save()
    try:
        try: current = importlib.metadata.version('xgboost')
        except importlib.metadata.PackageNotFoundError: current = None
        assert current is None or current == VERSION, 'Refuse replacing a different research dependency'
        state['prior_version'] = current
        state['installation_attempted'] = current is None; save()
        if current is None:
            env = {k:v for k,v in os.environ.items() if not k.startswith('PIP_')}
            env['PIP_CONFIG_FILE']='/dev/null';env['PYTHONNOUSERSITE']='1'
            report = ROOT/'notes/geobc_xgboost_dependency_v92_pip_report.json'
            log = ROOT/'notes/geobc_xgboost_dependency_v92_install.log'
            with log.open('x',encoding='utf-8') as stream:
                result = subprocess.run([sys.executable,'-m','pip','install','xgboost=='+VERSION,
                    '--only-binary=:all:','--index-url','https://pypi.org/simple',
                    '--timeout','20','--retries','0','--disable-pip-version-check','--no-input',
                    '--cache-dir',str(ROOT/'runs/dependency_cache_v92'),'--report',str(report)],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,180-(time.time()-began)))
            state['pip_exit_code']=result.returncode
            assert result.returncode == 0, 'Binary dependency install failed; retain log, no auto retry'
        assert time.time()-began < 180
        import xgboost
        assert xgboost.__version__ == VERSION
        build = xgboost.build_info();assert build['USE_CUDA'], 'CPU-only wheel is ineligible'
        lib = Path(xgboost.core._LIB._name)
        state.update(status='completed',build_info=build,installed_version=xgboost.__version__,
            library_path=str(lib),library_bytes=lib.stat().st_size,
            library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
            compiled_GPU_support=True,actual_training_GPU_verified=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    except BaseException as err:
        state.update(status='failed',error_type=type(err).__name__,error=str(err),automatic_retry=False)
        raise
    finally:
        state['elapsed_epoch_seconds']=time.time()-began;save()
    print(json.dumps({k:state[k] for k in ('status','installed_version','elapsed_epoch_seconds','compiled_GPU_support','actual_training_GPU_verified')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--install',action='store_true')
    if p.parse_args().install:prepare()
    else:print('{"status":"source_only","dependency_installs":0,"arrays_GT_GPU":0}')
