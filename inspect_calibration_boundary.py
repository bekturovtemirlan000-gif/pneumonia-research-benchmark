"""Describe saved CPU/GPU calibration differences without changing any decision rule."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def inspect(directory):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    frame=pd.read_csv(directory/'calibration_cpu_scores.csv',float_precision='round_trip')
    cutoff=summary['saved_threshold']
    saved=frame.probability.to_numpy();cpu=frame.cpu_frozen_temperature_score.to_numpy()
    if len(frame)!=summary['images'] or not np.isfinite(saved).all() or not np.isfinite(cpu).all():
        raise ValueError('Invalid saved scores')
    original_predictions=frame.prediction.to_numpy(dtype=int)
    changed=original_predictions!=(cpu>=cutoff)
    if int(changed.sum())!=summary['changed_frozen_calibration_predictions']:
        raise ValueError('Classification-change count differs from summary')
    rows=[]
    for row in frame.loc[changed].itertuples():
        rows.append(dict(sha256=row.sha256,label=int(row.y_true),saved_score=row.probability,
            cpu_score=row.cpu_frozen_temperature_score,saved_margin=row.probability-cutoff,
            cpu_margin=row.cpu_frozen_temperature_score-cutoff,
            saved_prediction=int(row.prediction),cpu_prediction=int(row.cpu_frozen_temperature_score>=cutoff)))
    return dict(image_size=summary['image_size'],images=len(frame),cutoff=cutoff,
        changed_images=rows,max_absolute_score_difference=float(np.max(np.abs(cpu-saved))),
        reread_saved_score_decisions_differ_from_stored=int(np.sum((saved>=cutoff)!=original_predictions)),
        saved_score_distance_to_cutoff_percentiles=np.percentile(np.abs(saved-cutoff),[0,1,5,50]).tolist(),
        saved_images_within_1e_4_of_cutoff=int(np.sum(np.abs(saved-cutoff)<=1e-4)),
        strict_verification_status=summary['status'],frozen_config_changed=False,
        interpretation='Post hoc numerical-discrepancy inspection on calibration images only. Does not relax the original zero-change criterion, change the cutoff, or establish clinical validity.')


if __name__=='__main__':
    root=Path('/kaggle/input')
    matches=[p for p in root.rglob('calibration_reproduction_outputs') if p.is_dir()]
    if len(matches)!=1: raise ValueError('Attach exactly one completed calibration reproduction output')
    out=Path('/kaggle/working/calibration_boundary_outputs');out.mkdir(exist_ok=True)
    for size in (224,320):
        report=inspect(matches[0]/str(size))
        (out/f'{size}.json').write_text(json.dumps(report,indent=2))
        print('CALIBRATION_BOUNDARY',json.dumps(report),flush=True)
