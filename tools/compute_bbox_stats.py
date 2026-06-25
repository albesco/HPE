import json,collections,sys
from pathlib import Path

candidates=[
    Path('data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_test.json'),
    Path('data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_val.json'),
    Path('data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_train.json')
]
ann_path=None
for p in candidates:
    if p.exists():
        ann_path=p
        break
if ann_path is None:
    print('ANNOTATION_FILE_NOT_FOUND', file=sys.stderr)
    sys.exit(2)
ann=json.loads(ann_path.read_text())
counts=collections.Counter()
for ann_item in ann.get('annotations',[]):
    bbox=ann_item.get('bbox')
    if not bbox or len(bbox)<4: continue
    w=int(round(bbox[2])); h=int(round(bbox[3]))
    counts[(w,h)]+=1
total=sum(counts.values())
if total==0:
    print('NO_BBOXES_FOUND')
    sys.exit(1)
most=counts.most_common(20)
print('ANN_PATH:', ann_path)
print('TOTAL_BBOXES:', total)
for i,(wh,cnt) in enumerate(most[:3],1):
    print(f'TOP{i}: size={wh[0]}x{wh[1]}, count={cnt}, pct={cnt/total*100:.2f}%')
print('\nNEXT_MOST_COMMON:')
for wh,cnt in most[3:10]:
    print(f'size={wh[0]}x{wh[1]}, count={cnt}, pct={cnt/total*100:.2f}%')
