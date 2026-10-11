"""Restore only frozen source line endings after a Git LF checkout."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MAP=ROOT/'notes/geobc_frozen_runtime_source_bytes.json'


def sha(value):return hashlib.sha256(value).hexdigest()


def run(restore=False):
    manifest=json.loads(MAP.read_text(encoding='utf-8-sig'));pending=[]
    for name,item in manifest['files'].items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        raw=path.read_bytes()
        if sha(raw)==item['frozen_runtime_sha256']:continue
        canonical=raw.replace(b'\r\n',b'\n')
        assert sha(canonical)==item['canonical_LF_sha256'],'Source content differs: '+name
        candidates=(canonical,canonical.replace(b'\n',b'\r\n'))
        valid=[c for c in candidates if sha(c)==item['frozen_runtime_sha256']]
        assert valid,'Frozen mixed line ending or unknown source bytes: '+name
        pending.append((path,valid[0]))
    # Validate every file before writing any. No content substitutions or pin changes.
    if restore:
        for path,content in pending:
            temporary=path.with_suffix(path.suffix+'.geobc_line_endings.tmp')
            assert not temporary.exists()
            with temporary.open('xb') as stream:stream.write(content)
            temporary.replace(path)
    print(dict(status='frozen_runtime_bytes_match' if not pending or restore else 'line_endings_restore_needed',
               checked_files=len(manifest['files']),pending_files=len(pending),writes=len(pending) if restore else 0,
               GT_reads=0,GPU_calls=0,source_content_changed=False,scientific_pins_changed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    if args.check or args.restore:run(args.restore)
