"""Query a locally prepared compact response from JSONL physical coordinates."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
from pathlib import Path
import signal


def run(args):
    import torch
    from geobc.deployment import PreparedResponseSession,handle_line
    from geobc.runtime import GPUQueue
    from geobc.physics_inputs import local_path
    catalog=json.loads(Path(args.catalog).read_text());memory=30500000000
    def guard():
        if max(time.time()-ENTRY_EPOCH,time.monotonic()-ENTRY_MONOTONIC)>=args.whole_seconds:raise TimeoutError('Query session deadline')
        if torch.cuda.is_initialized():
            free,total=torch.cuda.mem_get_info()
            if total-free>memory:raise MemoryError('Query session memory budget')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Query session interrupted')))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with GPUQueue(args.lock):
        torch.cuda.set_per_process_memory_fraction(memory/torch.cuda.get_device_properties(0).total_memory)
        session=PreparedResponseSession(catalog,guard)
        with local_path(args.requests).open() as requests,Path(args.output).open('x',encoding='utf-8') as output:
            for line in requests:
                guard();output.write(json.dumps(handle_line(session,line),allow_nan=False)+'\n');output.flush()
        guard()
        print(json.dumps(dict(counts=session.counts,elapsed_seconds=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['catalog','requests','output','lock']:p.add_argument('--'+name,required=True)
    p.add_argument('--whole-seconds',type=float,default=900)
    run(p.parse_args())
