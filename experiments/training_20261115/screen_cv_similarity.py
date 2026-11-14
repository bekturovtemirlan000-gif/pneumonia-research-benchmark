"""Post hoc pHash screening across the actual exported outer folds; no model changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ['sha256', 'patient', 'y_true', 'fold']
FIELDS = ['pair_id', 'fold_a', 'fold_b', 'sha256_a', 'sha256_b', 'label_a', 'label_b', 'distance']
TRAIN_DIGEST = '293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_manifest(frame):
    if len(frame) != 3815 or frame.sha256.duplicated().any() or frame.patient.nunique() != 2153:
        raise ValueError('Wrong original training cohort')
    if frame.isna().any().any() or not frame.sha256.str.fullmatch('[0-9a-f]{64}').all():
        raise ValueError('Invalid image identities')
    if not frame.sha256.is_monotonic_increasing or hashlib.sha256('\n'.join(frame.sha256).encode()).hexdigest() != TRAIN_DIGEST:
        raise ValueError('Original training digest changed')
    if set(frame.fold) != {0, 1, 2} or not set(frame.y_true).issubset({0, 1}):
        raise ValueError('Invalid fold or label')
    if frame.groupby('patient').fold.nunique().max() != 1:
        raise ValueError('Proxy groups cross outer folds')
    if not frame.phash.str.fullmatch('[0-7][0-9a-f]{15}').all():
        raise ValueError('Invalid 64-bit pHash or nonzero DC bit')
    for fold, record in read(HERE / 'cv_folds.json').items():
        part = frame.loc[frame.fold == int(fold)]
        observed = dict(images=len(part), groups=int(part.patient.nunique()),
                        normal=int((part.y_true == 0).sum()), pneumonia=int((part.y_true == 1).sum()))
        if observed != record: raise ValueError('Fold membership counts changed')


def screen(frame, maximum_distance=6):
    """Screen every cross-fold pair; independently check all decisions with int.bit_count."""
    pieces = {fold: frame.loc[frame.fold == fold].sort_values('sha256').reset_index(drop=True)
              for fold in [0, 1, 2]}
    found = []; expected_distances = {}; comparisons = 0
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        left, right = pieces[a], pieces[b]
        right_values = np.array([int(v, 16) for v in right.phash], dtype=np.uint64)
        right_rows = list(right.itertuples())
        comparisons += len(left) * len(right)
        for l in left.itertuples():
            distances = np.bitwise_count(np.bitwise_xor(np.uint64(int(l.phash, 16)), right_values))
            for j in np.flatnonzero(distances <= maximum_distance):
                r = right_rows[int(j)]
                identifier = hashlib.sha256((l.sha256 + ':' + r.sha256).encode()).hexdigest()
                found.append(dict(pair_id=identifier, fold_a=a, fold_b=b, sha256_a=l.sha256,
                                  sha256_b=r.sha256, label_a=int(l.y_true), label_b=int(r.y_true),
                                  distance=int(distances[j])))
            # Independent exhaustive scalar implementation also checks rejected pairs.
            for r in right_rows:
                distance = (int(l.phash, 16) ^ int(r.phash, 16)).bit_count()
                if distance <= maximum_distance:
                    expected_distances[(l.sha256, r.sha256)] = distance
    pairs = pd.DataFrame(found, columns=FIELDS).sort_values(['fold_a', 'fold_b', 'sha256_a', 'sha256_b']).reset_index(drop=True)
    observed_distances = {(r.sha256_a, r.sha256_b): r.distance for r in pairs.itertuples()}
    if len(observed_distances) != len(pairs) or observed_distances != expected_distances or pairs.pair_id.duplicated().any():
        raise ValueError('Vectorized and exhaustive scalar screens differ')
    return pairs, comparisons


def write_csv(frame, path):
    Path(path).write_bytes(frame.to_csv(index=False, lineterminator='\n').encode('utf-8'))


def build(fingerprints):
    plan = read(HERE / 'cv_similarity_screen_plan.json')
    if hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest() != plan['screen_source_sha256_lf']:
        raise ValueError('Planned screen source changed')
    if plan['maximum_hamming_distance'] != 6 or sha(fingerprints) != plan['fingerprint_table_sha256']:
        raise ValueError('Fingerprint source or fixed distance rule changed')
    summary = read(HERE / 'cv_summary.json')
    if hashlib.sha256(json.dumps(summary, indent=2).encode()).hexdigest() != plan['cv_summary_sha256_lf']:
        raise ValueError('Wrong completed CV summary')
    cpu = read(HERE / 'cv_cpu_verification.json')
    if cpu['summary_sha256'] != plan['cv_summary_sha256_lf']:
        raise ValueError('CPU export concerns a different CV run')
    cohorts = []; prediction_inputs = []
    for recipe in ['b0_standard', 'b0_regularized', 'v2s_regularized', 'geometry']:
        meta = cpu['public_prediction_exports'][recipe]
        path = HERE / 'predictions' / meta['filename']
        if sha(path) != meta['sha256']: raise ValueError('Prediction export bytes changed')
        cohort = pd.read_csv(path, dtype={'sha256': str, 'patient': str}, usecols=KEYS)[KEYS]
        if len(cohort) != 3815 or cohort.sha256.duplicated().any(): raise ValueError('Invalid OOF coverage')
        if cohorts and not cohort.equals(cohorts[0]): raise ValueError('Recipes have different outer membership')
        cohorts.append(cohort)
        prediction_inputs.append(dict(recipe=recipe, filename=meta['filename'], sha256=meta['sha256']))
    original = pd.read_csv(fingerprints, dtype={'sha256': str, 'patient': str, 'phash': str})
    if len(original) != 5371 or original.sha256.duplicated().any(): raise ValueError('Invalid source fingerprint coverage')
    train = original.loc[original.split == 'train'].sort_values('sha256').reset_index(drop=True)
    identities = train[['sha256', 'patient', 'label']].rename(columns={'label': 'y_true'})
    if not identities.equals(cohorts[0][['sha256', 'patient', 'y_true']]):
        raise ValueError('Fingerprints differ from verified training identities or labels')
    frame = cohorts[0].copy(); frame['phash'] = train.phash.to_numpy()
    validate_manifest(frame)
    pairs, comparisons = screen(frame)
    public_input = HERE / 'cv_similarity_fingerprints.csv'
    public_pairs = HERE / 'cv_similarity_candidate_pairs.csv'
    write_csv(frame, public_input); write_csv(pairs, public_pairs)
    hashes = set(pairs.sha256_a) | set(pairs.sha256_b)
    involved = frame.loc[frame.sha256.isin(hashes)]
    report = dict(status='completed', analysis_type='Post hoc unadjudicated similarity screening',
                  completed_utc=datetime.now(timezone.utc).isoformat(), images=len(frame),
                  groups=int(frame.patient.nunique()), pairwise_comparisons=comparisons,
                  candidate_pairs=len(pairs), unique_candidate_images=len(hashes),
                  candidate_proxy_groups=int(involved.patient.nunique()),
                  fold_pair_counts={f'{a}-{b}': int(((pairs.fold_a == a) & (pairs.fold_b == b)).sum())
                                    for a, b in [(0, 1), (0, 2), (1, 2)]},
                  distance_counts={str(k): int((pairs.distance == k).sum()) for k in range(7)},
                  label_pair_counts={f'{a}-{b}': int(((pairs.label_a == a) & (pairs.label_b == b)).sum())
                                     for a in [0, 1] for b in [0, 1]},
                  fingerprint_table_sha256=plan['fingerprint_table_sha256'],
                  cv_summary_sha256_lf=plan['cv_summary_sha256_lf'], prediction_inputs=prediction_inputs,
                  public_fingerprints_sha256=sha(public_input), public_candidates_sha256=sha(public_pairs),
                  plan_sha256=sha(HERE / 'cv_similarity_screen_plan.json'),
                  executed_source_sha256_lf=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest(),
                  rule=plan['rule'], vectorized_and_exhaustive_scalar_results_agree=True,
                  model_scores_used_for_candidate_selection=False, new_visual_adjudication_performed=False,
                  cohort_or_model_changes=False, cpu_reproduction_status=cpu['status'],
                  environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  limitations=plan['limitations'])
    (HERE / 'cv_similarity_screen.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def recheck():
    report = read(HERE / 'cv_similarity_screen.json')
    if hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest() != report['executed_source_sha256_lf']:
        raise ValueError('Executed screen source changed')
    input_path = HERE / 'cv_similarity_fingerprints.csv'; pair_path = HERE / 'cv_similarity_candidate_pairs.csv'
    if sha(input_path) != report['public_fingerprints_sha256'] or sha(pair_path) != report['public_candidates_sha256']:
        raise ValueError('Published screen bytes changed')
    if sha(HERE / 'cv_similarity_screen_plan.json') != report['plan_sha256']:
        raise ValueError('Screen plan changed')
    frame = pd.read_csv(input_path, dtype={'sha256': str, 'patient': str, 'phash': str})
    validate_manifest(frame)
    recomputed, comparisons = screen(frame)
    saved = pd.read_csv(pair_path, dtype={'pair_id': str, 'sha256_a': str, 'sha256_b': str})
    if not saved.equals(recomputed) or comparisons != report['pairwise_comparisons'] or len(saved) != report['candidate_pairs']:
        raise ValueError('Candidate table does not reproduce')
    involved_hashes = set(saved.sha256_a) | set(saved.sha256_b)
    if len(involved_hashes) != report['unique_candidate_images'] or frame.loc[frame.sha256.isin(involved_hashes)].patient.nunique() != report['candidate_proxy_groups']:
        raise ValueError('Candidate image or proxy-group count differs')
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if int(((saved.fold_a == a) & (saved.fold_b == b)).sum()) != report['fold_pair_counts'][f'{a}-{b}']:
            raise ValueError('Fold-pair count differs')
    for distance in range(7):
        if int((saved.distance == distance).sum()) != report['distance_counts'][str(distance)]:
            raise ValueError('Distance count differs')
    for a in [0, 1]:
        for b in [0, 1]:
            if int(((saved.label_a == a) & (saved.label_b == b)).sum()) != report['label_pair_counts'][f'{a}-{b}']:
                raise ValueError('Label-pair count differs')
    print(json.dumps(dict(status='passed', candidate_pairs=len(saved), pairwise_comparisons=comparisons,
                         scope='Exact public-file hashes and exhaustive re-screening; no image or patient adjudication')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recheck', action='store_true')
    parser.add_argument('--fingerprints', type=Path, default=ROOT / 'artifacts/november_similarity_reproduction/fingerprints.csv')
    args = parser.parse_args()
    recheck() if args.recheck else build(args.fingerprints)
