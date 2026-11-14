"""Capture the separate grouped-CV log without overwriting primary results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from capture_log import parse

HERE = Path(__file__).resolve().parent


def capture(path):
    raw = Path(path).read_text(encoding='utf-8')
    parsed = parse(raw)
    names = ['CV_ENVIRONMENT', 'CV_FOLDS', 'CV_TRIAL_DONE', 'CV_FOLD_FROZEN', 'CV_FOLD_DONE', 'GROUPED_CV_DONE']
    collected = {name: [] for name in names}
    for line in raw.splitlines():
        row = re.match(r'^([0-9]+(?:\.[0-9]+)?)s\s+([0-9]+)\s+(.*)$', line)
        if not row: continue
        message = row[3]
        for name in names:
            if message.startswith(name + ' '):
                collected[name].append(json.loads(message[len(name) + 1:]))
                break
    for name in ['CV_ENVIRONMENT', 'CV_FOLDS', 'GROUPED_CV_DONE']:
        if len(collected[name]) > 1: raise ValueError('Duplicate CV terminal record: ' + name)
    trials = collected['CV_TRIAL_DONE']
    if len({r['trial'] for r in trials}) != len(trials): raise ValueError('Duplicate CV trial')
    mapping = dict(CV_ENVIRONMENT='cv_environment.json', CV_FOLDS='cv_folds.json', GROUPED_CV_DONE='cv_summary.json')
    for marker, filename in mapping.items():
        if collected[marker]: (HERE / filename).write_text(json.dumps(collected[marker][0], indent=2), encoding='utf-8')
    for marker, filename in [('CV_TRIAL_DONE','cv_training_records.json'), ('CV_FOLD_FROZEN','cv_frozen_folds.json'), ('CV_FOLD_DONE','cv_fold_results.json')]:
        if collected[marker]: (HERE / filename).write_text(json.dumps(collected[marker], indent=2), encoding='utf-8')
    if parsed['epochs']:
        (HERE / 'cv_epoch_history.json').write_text(json.dumps(parsed['epochs'], indent=2), encoding='utf-8')
    if collected['GROUPED_CV_DONE']:
        result = collected['GROUPED_CV_DONE'][0]
        if len(trials) != 27 or trials != result['training_records'] or collected['CV_FOLD_DONE'] != result['folds']:
            raise ValueError('Completed CV summary differs from individual logged results')
    progress = dict(captured_utc=datetime.now(timezone.utc).isoformat(), source='Visible Kaggle Logs table through Chrome',
                    log_sha256=hashlib.sha256(raw.encode()).hexdigest(), completed_trials=len(trials), planned_trials=27,
                    completed_fold_ensembles=len(collected['CV_FOLD_DONE']), logged_epoch_rows=len(parsed['epochs']),
                    last_epoch=parsed['epochs'][-1] if parsed['epochs'] else None,
                    complete_summary_observed=bool(collected['GROUPED_CV_DONE']),
                    scope='Compact observed logs; does not independently attest terminal success or complete artifact download')
    (HERE / 'cv_progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
    print(json.dumps(progress, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    capture(parser.parse_args().log)
