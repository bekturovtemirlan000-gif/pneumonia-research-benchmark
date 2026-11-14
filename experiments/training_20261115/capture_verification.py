"""Recover complete verification reports and hash-checked CSVs from visible logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

HERE = Path(__file__).resolve().parent


def capture(log, kind):
    raw = Path(log).read_text(encoding='utf-8')
    report_marker = 'NOVEMBER_CPU_VERIFICATION' if kind == 'primary' else 'GROUPED_CV_CPU_VERIFICATION'
    final = []; tables = {}; blocks = {}
    for line in raw.splitlines():
        row = re.match(r'^([0-9]+(?:\.[0-9]+)?)s\s+([0-9]+)\s+(.*)$', line)
        if not row: continue
        message = row[3]
        if message.startswith(report_marker + ' '): final.append(json.loads(message[len(report_marker)+1:]))
        if message.startswith('PUBLIC_PREDICTION_TABLE '):
            meta = json.loads(message[len('PUBLIC_PREDICTION_TABLE '):])
            if meta['filename'] in tables: raise ValueError('Duplicate table declaration')
            tables[meta['filename']] = meta
        if message.startswith('PUBLIC_PREDICTION_BLOCK '):
            block = json.loads(message[len('PUBLIC_PREDICTION_BLOCK '):])
            target = blocks.setdefault(block['filename'], {})
            if block['index'] in target: raise ValueError('Duplicate prediction block')
            target[block['index']] = block['rows']
    if len(final) != 1: raise ValueError('Exactly one complete verification report required')
    report = final[0]
    declared = {v['filename']: v for v in report['public_prediction_exports'].values()}
    if declared != tables or set(blocks) != set(tables): raise ValueError('Export declarations differ from final report')
    payloads = {}
    for name, meta in tables.items():
        if not re.fullmatch(r'public_[a-z0-9_]+\.csv', name): raise ValueError('Unexpected export filename')
        if sorted(blocks[name]) != list(range(meta['blocks'])): raise ValueError('Missing prediction blocks')
        records = [r for index in range(meta['blocks']) for r in blocks[name][index]]
        frame = pd.DataFrame.from_records(records, columns=meta['columns'])
        if len(frame) != meta['rows'] or frame.isna().any().any() or frame.sha256.duplicated().any():
            raise ValueError('Incomplete or duplicate prediction records')
        if not frame.sha256.str.fullmatch('[0-9a-f]{64}').all(): raise ValueError('Invalid image identifier')
        payload = frame.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8')
        if hashlib.sha256(payload).hexdigest() != meta['sha256']: raise ValueError('Prediction export hash mismatch: ' + name)
        payloads[name] = payload
    destination = HERE / 'predictions'; destination.mkdir(exist_ok=True)
    for name, payload in payloads.items(): (destination/name).write_bytes(payload)
    filename = 'cpu_verification.json' if kind == 'primary' else 'cv_cpu_verification.json'
    (HERE/filename).write_text(json.dumps(report, indent=2), encoding='utf-8')
    receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(), source='Visible Kaggle Logs table via Chrome',
                   kind=kind, verification_status=report['status'], log_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                   tables=list(tables.values()), all_export_byte_hashes_verified=True,
                   scope='Complete image-level score exports reconstructed from bounded JSON log blocks and matched to Kaggle CSV SHA256; source images and weights not included')
    (HERE/(kind+'_prediction_capture.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(kind=kind, verification_status=report['status'], tables=len(tables),
                         rows=sum(t['rows'] for t in tables.values()), all_byte_hashes_verified=True)))
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--kind',choices=['primary','cv'],required=True)
    args=parser.parse_args();capture(args.log,args.kind)
