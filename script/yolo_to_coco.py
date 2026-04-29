from pathlib import Path
import argparse
import json

from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection and export bboxes as COCO JSON."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--model", required=True, help="YOLO model weights path.")
    parser.add_argument("--output", required=True, help="Output COCO JSON path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    output_path = Path(args.output)

    image = Image.open(image_path)
    width, height = image.size

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        classes=[0],
        conf=args.conf,
        save=False,
        verbose=False,
    )

    annotations = []
    for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy(), start=1):
        x1, y1, x2, y2 = [float(v) for v in box]
        x = max(0.0, x1)
        y = max(0.0, y1)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        annotations.append(
            {
                "id": idx,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
            }
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {output_path}")
    print(f"Boxes: {len(annotations)}")


if __name__ == "__main__":
    main()
