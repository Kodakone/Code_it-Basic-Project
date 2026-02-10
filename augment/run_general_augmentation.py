import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm

# 레포 루트를 sys.path에 넣어서 dataloader import가 되게 함
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataloader.dataset_load import TRAIN_IMG_DIR, ANNOTATION_DIR
from dataloader.general_augmentation import build_general_augmentation

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_cv2(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return img


def write_image_cv2(path: Path, img_bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")


def ensure_xyxy_from_coco_bbox(bbox, img_w, img_h):
    # COCO bbox = [x, y, w, h]
    x, y, w, h = map(float, bbox)
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # clamp
    x1 = max(0.0, min(x1, img_w - 1.0))
    x2 = max(0.0, min(x2, img_w - 1.0))
    y1 = max(0.0, min(y1, img_h - 1.0))
    y2 = max(0.0, min(y2, img_h - 1.0))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def sanitize_bboxes_labels(bboxes_xyxy, cls_ids, img_w, img_h):
    clean_boxes, clean_cls = [], []
    for b, c in zip(bboxes_xyxy, cls_ids):
        if b is None or len(b) != 4:
            continue
        x1, y1, x2, y2 = map(float, b)
        # 최소 1픽셀 이상만 유지
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        # clamp
        x1 = max(0.0, min(x1, img_w - 1.0))
        x2 = max(0.0, min(x2, img_w - 1.0))
        y1 = max(0.0, min(y1, img_h - 1.0))
        y2 = max(0.0, min(y2, img_h - 1.0))

        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        clean_boxes.append([x1, y1, x2, y2])
        clean_cls.append(int(c))
    return clean_boxes, clean_cls


def xyxy_to_yolo_line(x1, y1, x2, y2, cls, img_w, img_h):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{int(cls)} {cx / img_w:.6f} {cy / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def save_yolo_label(label_path: Path, boxes_xyxy, cls_ids, img_w, img_h):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [xyxy_to_yolo_line(*b, c, img_w, img_h) for b, c in zip(boxes_xyxy, cls_ids)]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def build_targets_by_filename(train_img_dir: Path, annotation_dir: Path):
    """
    코랩의 targets_by_filename 로직 그대로:
    - train_images의 파일명 set을 만들고
    - train_annotations 아래 모든 COCO json을 순회하며
      images/annotations를 읽어서 filename -> {boxes, labels}를 누적
    """
    img_paths = sorted([p for p in train_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    train_img_files = [p.name for p in img_paths]
    train_img_set = set(train_img_files)

    json_files = list(annotation_dir.rglob("*.json"))

    targets_by_filename = defaultdict(lambda: {"boxes": [], "labels": []})
    scanned = 0
    used_json = 0

    for jp in tqdm(json_files, desc="SCAN COCO JSON"):
        scanned += 1
        try:
            with open(jp, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception:
            continue

        images = coco.get("images", [])
        anns = coco.get("annotations", [])

        if not images or not anns:
            continue

        imageid_to_fname = {}
        imageid_to_wh = {}

        for img in images:
            fname = img.get("file_name")
            if not fname:
                continue
            # train_images에 실제 존재하는 파일만
            if fname in train_img_set:
                iid = img.get("id")
                if iid is None:
                    continue
                imageid_to_fname[iid] = fname
                w = img.get("width")
                h = img.get("height")
                if w is not None and h is not None:
                    imageid_to_wh[iid] = (int(w), int(h))

        if not imageid_to_fname:
            continue

        used_json += 1

        for ann in anns:
            img_id = ann.get("image_id")
            if img_id not in imageid_to_fname:
                continue

            bbox = ann.get("bbox")
            cat_id = ann.get("category_id")
            if bbox is None or cat_id is None:
                continue

            fname = imageid_to_fname[img_id]

            # width/height가 json에 있으면 그걸 쓰고, 없으면 나중에 이미지에서 보정
            if img_id in imageid_to_wh:
                iw, ih = imageid_to_wh[img_id]
            else:
                # 임시 큰 값으로 넣고, 나중에 sanitize에서 실제 이미지 크기로 clamp됨
                iw, ih = 1_000_000, 1_000_000

            xyxy = ensure_xyxy_from_coco_bbox(bbox, iw, ih)
            targets_by_filename[fname]["boxes"].append(xyxy)
            targets_by_filename[fname]["labels"].append(int(cat_id))

    print("[INFO] train images:", len(train_img_files))
    print("[INFO] json scanned:", scanned)
    print("[INFO] json used (matched at least 1 image):", used_json)
    print("[INFO] images with annotations:", len(targets_by_filename))

    return img_paths, targets_by_filename


def main():
    OUT_ROOT = REPO_ROOT / "yolo_dataset_aug"
    OUT_IMG_DIR = OUT_ROOT / "images" / "train"
    OUT_LBL_DIR = OUT_ROOT / "labels" / "train"
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    aug_tf = build_general_augmentation()

    img_paths, targets_by_filename = build_targets_by_filename(TRAIN_IMG_DIR, ANNOTATION_DIR)

    AUG_N = 1  # 이미지당 증강본 개수
    saved = 0
    skipped = 0
    no_ann = 0

    print("[INFO] NOTE: 원본 이미지는 저장하지 않고, 증강본만 저장합니다.")
    print("[INFO] TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
    print("[INFO] ANNOTATION_DIR:", ANNOTATION_DIR)
    print("[INFO] OUT_ROOT:", OUT_ROOT)
    print("[INFO] AUG_N:", AUG_N)

    for img_path in tqdm(img_paths, desc="GENERAL AUG (AUG ONLY)"):
        fname = img_path.name
        if fname not in targets_by_filename:
            no_ann += 1
            continue

        img = read_image_cv2(img_path)
        h, w = img.shape[:2]

        bboxes_xyxy = targets_by_filename[fname]["boxes"]
        labels = targets_by_filename[fname]["labels"]

        bboxes_xyxy, labels = sanitize_bboxes_labels(bboxes_xyxy, labels, w, h)
        if len(bboxes_xyxy) == 0:
            skipped += 1
            continue

        stem = img_path.stem
        ext = img_path.suffix.lower()

        for k in range(AUG_N):
            transformed = aug_tf(image=img, bboxes=bboxes_xyxy, class_labels=labels)
            imgA = transformed["image"]
            bbA = transformed["bboxes"]
            clA = transformed["class_labels"]

            hA, wA = imgA.shape[:2]
            final_bboxes, final_labels = sanitize_bboxes_labels(bbA, clA, wA, hA)
            if len(final_bboxes) == 0:
                skipped += 1
                continue

            out_img = OUT_IMG_DIR / f"{stem}_aug{k+1}{ext}"
            out_lbl = OUT_LBL_DIR / f"{stem}_aug{k+1}.txt"

            # 덮어쓰기 방지
            if out_img.exists() or out_lbl.exists():
                continue

            write_image_cv2(out_img, imgA)
            save_yolo_label(out_lbl, final_bboxes, final_labels, wA, hA)
            saved += 1

    print("[DONE] saved:", saved)
    print("[DONE] skipped:", skipped)
    print("[DONE] no_annotation:", no_ann)


if __name__ == "__main__":
    main()
