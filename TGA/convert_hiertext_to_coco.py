
import json
import os
import cv2
import argparse

def convert(json_path, img_dir, out_path):
    print("Loading:", json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for entry in data["annotations"]:
        image_name = entry["image_id"] + ".jpg"
        image_path = os.path.join(img_dir, image_name)

        if not os.path.exists(image_path):
            print("Missing image:", image_path)
            continue

        # Read image to get size
        img = cv2.imread(image_path)
        if img is None:
            print("Corrupt image:", image_path)
            continue

        h, w = img.shape[:2]

        # Build image entry
        images.append({
            "id": img_id,
            "file_name": image_name,
            "height": h,
            "width": w
        })

        # Each paragraph = one segmentation
        for para in entry["paragraphs"]:
            vertices = para["vertices"]
            seg = []
            for x, y in vertices:
                seg.extend([int(x), int(y)])

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [seg],
                "iscrowd": 0,
                "bbox": [],           # optional (not needed for your model)
                "area": 0             # optional (not needed)
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "text"}]
    }

    with open(out_path, "w") as f:
        json.dump(coco, f)

    print("DONE:", out_path)
    print("# Images:", len(images))
    print("# Annotations:", len(annotations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl")
    parser.add_argument("--img-dir")
    parser.add_argument("--out")
    args = parser.parse_args()

    convert(args.jsonl, args.img_dir, args.out)


