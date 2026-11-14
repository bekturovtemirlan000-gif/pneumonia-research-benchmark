"""Emit de-identified public-dataset prediction tables through bounded log blocks."""
import hashlib
import json
from pathlib import Path
import re


def export(frame, output, filename):
    if not re.fullmatch(r'[a-z0-9_]+\.csv', filename): raise ValueError('Invalid export filename')
    if frame.isna().any().any() or frame.sha256.duplicated().any(): raise ValueError('Invalid prediction table')
    frame = frame.sort_values('sha256').reset_index(drop=True)
    payload = frame.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8')
    path = Path(output) / filename
    path.write_bytes(payload)
    records = frame.to_dict('records'); block_size = 50
    meta = dict(filename=filename, rows=len(frame), columns=frame.columns.tolist(),
                sha256=hashlib.sha256(payload).hexdigest(), blocks=(len(frame)+block_size-1)//block_size,
                format='UTF-8 CSV, LF, no index, floats %.17g',
                scope='Public research-dataset pixel hashes, filename proxy groups, inherited labels and model scores; no image pixels or verified patient identity')
    print('PUBLIC_PREDICTION_TABLE', json.dumps(meta), flush=True)
    for index in range(meta['blocks']):
        print('PUBLIC_PREDICTION_BLOCK', json.dumps(dict(filename=filename, index=index,
              rows=records[index*block_size:(index+1)*block_size]), allow_nan=False), flush=True)
    return meta
