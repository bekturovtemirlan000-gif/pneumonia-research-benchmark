"""Extract compact evidence from text copied from Kaggle's visible Logs table."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
MARKERS = ['ENVIRONMENT', 'DATA_AUDIT', 'TRIAL_DONE', 'DEVELOPMENT_CHOICE', 'MODEL_FROZEN',
           'COMPARISON_DONE', 'TRAINING_EXPERIMENT_DONE', 'NOVEMBER_CPU_VERIFICATION']


def parse(log):
    records = {name: [] for name in MARKERS}
    epochs = {}
    audits = []
    for line in log.splitlines():
        # Rendered Kaggle table uses elapsed-seconds, row-number, and message columns.
        row = re.match(r'^([0-9]+(?:\.[0-9]+)?)s\s+([0-9]+)\s+(.*)$', line)
        if not row:
            continue
        seconds, number, message = float(row[1]), int(row[2]), row[3]
        if message.startswith('EPOCH '):
            _, trial, payload = message.split(' ', 2)
            value = json.loads(payload)
            key = (trial, value['epoch'])
            value = dict(value, trial=trial, kaggle_elapsed_seconds=seconds)
            if key in epochs and epochs[key] != value:
                raise ValueError('Conflicting epoch rows')
            epochs[key] = value
        elif message.startswith('AUDIT '):
            _, done, total = message.split()
            audits.append(dict(images_verified=int(done), images_total=int(total), kaggle_elapsed_seconds=seconds))
        else:
            for marker in MARKERS:
                if message.startswith(marker + ' '):
                    value = json.loads(message[len(marker) + 1:])
                    records[marker].append(dict(kaggle_elapsed_seconds=seconds, row=number, value=value))
                    break
    for name, values in records.items():
        if name != 'TRIAL_DONE' and len(values) > 1:
            raise ValueError('Multiple final records: ' + name)
    return dict(records=records, epochs=list(epochs.values()), audits=audits)


def capture(log_path):
    log_path = Path(log_path)
    raw = log_path.read_text(encoding='utf-8')
    parsed = parse(raw)
    records = parsed['records']
    captured_utc = datetime.now(timezone.utc).isoformat()
    provenance = dict(captured_utc=captured_utc, source='Visible Kaggle Logs table copied through authenticated Chrome',
                      log_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                      scope='Compact log evidence; not a downloaded complete model or image-level prediction archive')
    mapping = dict(ENVIRONMENT='environment.json', DATA_AUDIT='data_audit.json', DEVELOPMENT_CHOICE='development_choice.json',
                   MODEL_FROZEN='frozen_model.json', COMPARISON_DONE='comparison.json',
                   TRAINING_EXPERIMENT_DONE='summary.json', NOVEMBER_CPU_VERIFICATION='cpu_verification.json')
    for marker, filename in mapping.items():
        if records[marker]:
            (HERE / filename).write_text(json.dumps(records[marker][0]['value'], indent=2), encoding='utf-8')
    trials = [v['value'] for v in records['TRIAL_DONE']]
    if len({t['trial'] for t in trials}) != len(trials):
        raise ValueError('Duplicate completed trial')
    if trials:
        (HERE / 'development_trials.json').write_text(json.dumps(trials, indent=2), encoding='utf-8')
    if parsed['epochs']:
        (HERE / 'epoch_history.json').write_text(json.dumps(parsed['epochs'], indent=2), encoding='utf-8')
    complete = bool(records['TRAINING_EXPERIMENT_DONE'])
    if complete:
        result = records['TRAINING_EXPERIMENT_DONE'][0]['value']
        if len(trials) != 9 or trials != result['trials']:
            raise ValueError('Completed summary differs from individual trial log records')
    progress = dict(provenance=provenance, training_summary_observed=complete, completed_trials=len(trials),
                    logged_epoch_rows=len(parsed['epochs']), last_epoch=parsed['epochs'][-1] if parsed['epochs'] else None,
                    last_audit=parsed['audits'][-1] if parsed['audits'] else None,
                    disclaimer='Observed log records do not by themselves establish the Kaggle terminal success status')
    (HERE / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
    print(json.dumps(progress, indent=2))
    return progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    capture(parser.parse_args().log)
