import json,collections,sys
from pathlib import Path
out=Path('/tmp/bbox_bucket_stats.txt')
ann_path=Path('data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_test.json')
if not ann_path.exists():
    print('ANNOTATION_FILE_NOT_FOUND', file=sys.stderr); sys.exit(2)
ann=json.loads(ann_path.read_text())
counts32=collections.Counter()
counts_standard=collections.Counter()
standard_sizes=[128,192,256,320,384,448,512,640,768,896,1024,1280]

def nearest_standard(x):
    return min(standard_sizes, key=lambda s: abs(s-x))

for ann_item in ann.get('annotations',[]):
    bbox=ann_item.get('bbox')
    if not bbox or len(bbox)<4: continue
    w=int(round(bbox[2])); h=int(round(bbox[3]))
    rw=(int(round(w/32.0))*32) if w>0 else 0
    rh=(int(round(h/32.0))*32) if h>0 else 0
    counts32[(rw,rh)]+=1
    counts_standard[(nearest_standard(w), nearest_standard(h))]+=1

with out.open('w') as f:
    total=sum(counts32.values())
    f.write(f'ANN_PATH: {ann_path}\n')
    f.write(f'TOTAL_BBOXES: {total}\n\n')
    f.write('Top-10 (multipli di 32):\n')
    for i,((w,h),cnt) in enumerate(counts32.most_common(10),1):
        f.write(f'{i}: {w}x{h} — count={cnt} — {cnt/total*100:.2f}%\n')
    f.write('\nTop-10 (dimensioni standard nearest):\n')
    for i,((w,h),cnt) in enumerate(counts_standard.most_common(10),1):
        f.write(f'{i}: {w}x{h} — count={cnt} — {cnt/total*100:.2f}%\n')
    small=med=large=0
    for (w,h),cnt in counts32.items():
        area=w*h
        if area<32*32: small+=cnt
        elif area<96*96: med+=cnt
        else: large+=cnt
    f.write(f'\nBy area buckets (coarse): small(<32^2)={small}, medium(32^2-96^2)={med}, large(>96^2)={large}\n')
print('WROTE', out)
