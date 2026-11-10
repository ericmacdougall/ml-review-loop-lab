"""Preserve raw evidence and provide strict JSON for publication."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'ml-loop-lab/results'


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [clean(x) for x in value]
    if isinstance(value,dict):
        result={k:clean(v) for k,v in value.items()}
        if value.get('flagged')==0 and 'precision' in value:
            result['precision']=None
            if 'precision_ci' in value:result['precision_ci']=None
        return result
    return value


def main():
    target=RESULTS/'public'
    target.mkdir(exist_ok=True)
    for name in ['summary.json','signals.json','peek-demo.json']:
        value=json.loads((RESULTS/name).read_text())
        (target/name).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    raw=(RESULTS/'final.jsonl').read_bytes()
    (RESULTS/'raw-final.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    fields={'original_prediction_sha256':hashlib.sha256(raw).hexdigest(),
            'original_prediction_lines':len(raw.splitlines()),
            'frozen_source_commit':'d50ac317877f056103ca4257b1e719c530d95d5e',
            'normalization':'Non-finite values and empty-set precision become null only in public JSON. Original raw evidence is preserved unchanged.',
            'artifacts':{}}
    for p in sorted([RESULTS/'raw-final.jsonl.gz',*target.glob('*.json')]):
        fields['artifacts'][str(p.relative_to(RESULTS))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RESULTS/'evidence-manifest.json').write_text(json.dumps(fields,indent=2)+'\n')
    print('Preserved300 raw runs; exported strict JSON and evidence hashes.')


if __name__=='__main__':main()
