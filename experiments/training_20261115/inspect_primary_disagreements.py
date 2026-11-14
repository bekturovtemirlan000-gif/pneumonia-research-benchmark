"""Post hoc descriptive pairing of the two already-frozen primary decisions."""
from datetime import datetime, timezone
import json
import numpy as np
from audit_prediction_tables import HERE, read, table


def inspect():
    old = read('comparator_export.json')
    current = read('cpu_verification.json')
    a = table(old['public_prediction_exports']['original_320'], 618, 426)
    b = table(current['public_prediction_exports']['test'], 618, 426)
    if not a[['sha256', 'patient', 'y_true']].equals(b[['sha256', 'patient', 'y_true']]):
        raise ValueError('Frozen test identities differ')
    correct_a, correct_b = a.prediction == a.y_true, b.prediction == b.y_true
    counts = dict(both_correct=int(np.sum(correct_a & correct_b)),
                  original_only_correct=int(np.sum(correct_a & ~correct_b)),
                  november_only_correct=int(np.sum(~correct_a & correct_b)),
                  neither_correct=int(np.sum(~correct_a & ~correct_b)))
    if sum(counts.values()) != 618: raise ValueError('Incomplete pairing')
    classes = {}
    for label, name in [(0, 'normal'), (1, 'pneumonia')]:
        selected = a.y_true == label
        classes[name] = dict(images=int(selected.sum()),
                             decisions_differ=int(np.sum(selected & (a.prediction != b.prediction))),
                             original_only_correct=int(np.sum(selected & correct_a & ~correct_b)),
                             november_only_correct=int(np.sum(selected & ~correct_a & correct_b)))
    report = dict(status='completed', analysis_timing='Post hoc, after primary source-test outcomes were observed',
                  recorded_utc=datetime.now(timezone.utc).isoformat(), images=618, proxy_groups=426,
                  paired_correctness=counts, by_inherited_class=classes,
                  shared_false_negatives=int(np.sum((a.y_true == 1) & (a.prediction == 0) & (b.prediction == 0))),
                  source_table_sha256={meta['filename']:meta['sha256'] for meta in [
                      old['public_prediction_exports']['original_320'], current['public_prediction_exports']['test']]},
                  configuration_changed=False, fitting=False,
                  interpretation='Describes disagreements under the two original frozen cutoffs. No new selection, significance test, clinical adjudication or external claim.')
    (HERE/'primary_frozen_disagreements.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__': inspect()
