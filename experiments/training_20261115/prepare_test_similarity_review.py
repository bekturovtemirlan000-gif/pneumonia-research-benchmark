"""Render every source-test similarity candidate for nonclinical visual review."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def prepare(archive):
    source = ROOT / 'artifacts/november_similarity_reproduction/candidate_pairs.csv'
    reference = json.loads((HERE / 'similarity_reproduction.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference['candidate_table_sha256']:
        raise ValueError('Reproduced candidate table changed')
    frame = pd.read_csv(source)
    frame = frame.loc[(frame.split_a == 'test') | (frame.split_b == 'test')]
    frame = frame.sort_values(['normalized_pixel_correlation', 'pair_id'], ascending=[False, True]).reset_index(drop=True)
    if len(frame) != 269 or frame.pair_id.duplicated().any():
        raise ValueError('Expected all 269 unique test-involving candidates')
    plan = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                purpose='Expanded AI-assisted nonclinical image-correspondence review while grouped training runs',
                selection='Every reproduced pHash-screened pair involving the retained source-test cohort; 269 pairs, ordered by descending pixel correlation then pair ID',
                candidate_table_sha256=reference['candidate_table_sha256'],
                pair_ids=frame.pair_id.tolist(), pairs_per_sheet=6, image_panel_pixels=[540,245],
                outcomes_known=True,
                scope='Post hoc descriptive data review. No new image exclusions, model fitting, threshold fitting, diagnostic adjudication or patient-identity inference.',
                interpretation='Thumbnail correspondence inspection cannot exclude transformed copies or repeated acquisitions; potential matches require original-resolution independent adjudication.')
    (HERE/'test_similarity_review_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    output = ROOT / 'artifacts/november_test_similarity_review'
    output.mkdir(parents=True,exist_ok=True)
    sheets = []; thumbnails = {}
    with zipfile.ZipFile(archive) as z:
        for page in range((len(frame)+5)//6):
            rows = frame.iloc[page*6:(page+1)*6]
            sheet = Image.new('RGB',(1300,len(rows)*290),'white'); draw=ImageDraw.Draw(sheet)
            for index,row in enumerate(rows.itertuples()):
                y=index*290
                draw.text((8,y+5),f'{page*6+index+1:03d} | {row.pair_id} | {row.split_a}/{row.split_b} | pHash {row.distance} | r={row.normalized_pixel_correlation:.6f}',fill='black')
                for col,side in enumerate(['a','b']):
                    name=getattr(row,'path_'+side); expected=getattr(row,'sha256_'+side)
                    if expected not in thumbnails:
                        with Image.open(BytesIO(z.read(name))) as image:
                            gray=image.convert('L')
                            if hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()!=expected:
                                raise ValueError('Decoded image identity changed')
                            thumbnails[expected]=ImageOps.pad(image.convert('RGB'),(540,245),method=Image.Resampling.LANCZOS,color='black')
                    sheet.paste(thumbnails[expected],(col*650+8,y+37))
                    draw.text((col*650+8,y+22),Path(name).name,fill='black')
            path=output/f'test_pairs_{page+1:02d}.png'; sheet.save(path)
            sheets.append(dict(sheet=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pair_ids=rows.pair_id.tolist()))
    (output/'sheet_manifest.json').write_text(json.dumps(dict(plan=plan,sheets=sheets),indent=2),encoding='utf-8')
    print(json.dumps(dict(pairs=len(frame),sheets=len(sheets),unique_decoded_images=len(thumbnails),output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    prepare(parser.parse_args().archive)
