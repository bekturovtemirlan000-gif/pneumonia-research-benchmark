"""Check recorded visual-review coverage without generating review decisions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def finalize():
    plan=json.loads((HERE/'test_similarity_review_plan.json').read_text())
    observation_path=HERE/'test_similarity_review_observations.json'
    observations=json.loads(observation_path.read_text())
    local=ROOT/'artifacts/november_test_similarity_review'
    manifest=json.loads((local/'sheet_manifest.json').read_text())
    if manifest['plan']!=plan: raise ValueError('Rendered review plan changed')
    notes=observations['notes']; sheets=manifest['sheets']
    if [n['sheet'] for n in notes]!=list(range(1,46)) or len(sheets)!=45:
        raise ValueError('Incomplete sheet review')
    rows=[r for n in notes for r in n['observed_pairs']]
    if rows!=list(range(1,270)): raise ValueError('Incomplete or duplicate pair review')
    records=[]
    for note,sheet in zip(notes,sheets):
        if hashlib.sha256((local/sheet['sheet']).read_bytes()).hexdigest()!=sheet['sha256']:
            raise ValueError('Rendered sheet changed')
        ids=[plan['pair_ids'][i-1] for i in note['observed_pairs']]
        if ids!=sheet['pair_ids'] or not set(note['possible_copy_rows'])<=set(note['observed_pairs']):
            raise ValueError('Review does not match rendered pair order')
        for row,pair_id in zip(note['observed_pairs'],ids):
            records.append(dict(row=row,pair_id=pair_id,sheet=note['sheet'],
                observation='possible_copy_requires_independent_review' if row in note['possible_copy_rows'] else 'copy_not_established_at_displayed_resolution'))
    previous=json.loads((HERE/'similarity_visual_review.json').read_text())
    prior=set(previous['reviewed_pair_ids']); current=set(plan['pair_ids'])
    result=dict(status='completed_nonclinical_thumbnail_review',reviewed_utc=datetime.now(timezone.utc).isoformat(),
        reviewer=observations['reviewer'],selection=plan['selection'],reviewed_pairs=269,reviewed_sheets=45,
        image_panel_pixels=plan['image_panel_pixels'],confirmed_transformed_copies=0,
        possible_copy_rows=[r['row'] for r in records if r['observation'].startswith('possible_')],
        observation='All displayed pairs had visible differences in image framing, body positioning, markers, overlays or internal patterns. No transformed copy was established. Differences in appearance do not prove different patients.',
        earlier_review_overlap=len(prior&current),unique_pairs_across_both_reviews=len(prior|current),
        total_screen_candidates=814,screen_candidates_not_visually_reviewed=814-len(prior|current),
        outcomes_known_before_review=True,cohort_changes=False,model_or_threshold_changes=False,
        diagnostic_adjudication=False,verified_patient_identity=False,independent_expert_review=False,
        original_resolution_registration=False,
        limitations=['AI-assisted thumbnail review cannot conclusively exclude transformed copies, repeated acquisitions or shared patients',
                     'Review covers all test-involving screen candidates, not all possible cross-partition image pairs',
                     'The fixed pHash screen can miss transformations; lower pixel correlation does not by itself exclude duplication',
                     'Post hoc data description does not repair prior source-test exposure or establish clinical validity'],
        image_publication='Local review panels are excluded from the source release; only code, hashes and observations are published',
        candidate_table_sha256=plan['candidate_table_sha256'],
        observations_sha256=hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        rendered_sheets=sheets,records=records)
    (HERE/'test_similarity_visual_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','reviewed_pairs','reviewed_sheets','earlier_review_overlap','unique_pairs_across_both_reviews','screen_candidates_not_visually_reviewed','possible_copy_rows']}))


if __name__=='__main__': finalize()
