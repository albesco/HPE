import json,collections,sys
from pathlib import Path
res_path=Path('data/output/experiments/YoloVitPose_mAP/test_20260522_151805/yolo_vitpose_keypoints_results.json')
if not res_path.exists():
    print('RESULTS_JSON_NOT_FOUND', file=sys.stderr); sys.exit(2)
results=json.loads(res_path.read_text())
counts32=collections.Counter(); counts_standard=collections.Counter()
standard_sizes=[128,192,256,320,384,448,512,640,768,896,1024,1280]

def nearest_standard(x):
    return min(standard_sizes, key=lambda s: abs(s-x))

for rec in results:
    bbox=rec.get('bbox')
    if not bbox or len(bbox)<4: continue
    w=int(round(bbox[2])); h=int(round(bbox[3]))
    rw=(int(round(w/32.0))*32) if w>0 else 0
    rh=(int(round(h/32.0))*32) if h>0 else 0
    counts32[(rw,rh)]+=1
    counts_standard[(nearest_standard(w), nearest_standard(h))]+=1

total=sum(counts32.values())
print('RESULTS_JSON:', res_path)
print('TOTAL_PREDICTIONS:', total)
print('\nTop-5 (multipli di 32):')
for i,((w,h),cnt) in enumerate(counts32.most_common(5),1):
    print(f'{i}: {w}x{h} — count={cnt} — {cnt/total*100:.2f}%')
print('\nTop-5 (standard sizes nearest):')
for i,((w,h),cnt) in enumerate(counts_standard.most_common(5),1):
    print(f'{i}: {w}x{h} — count={cnt} — {cnt/total*100:.2f}%')
