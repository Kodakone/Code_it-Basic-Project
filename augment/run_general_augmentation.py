import json
import sys
from collections import defaultdict
from pathlib import Path

# 레포 루트를 sys.path에 먼저 추가
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataloader.dataset_load import (  # noqa: E402
    ANNOTATION_DIR,
    CACHE_DIR,
    DATA_ROOT,
    TRAIN_IMG_DIR,
)
from dataloader.general_augmentation import build_general_augmentation  # noqa: E402

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
    lines = [
        xyxy_to_yolo_line(*b, c, img_w, img_h) for b, c in zip(boxes_xyxy, cls_ids)
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_ROOT = Path(DATA_ROOT).parent / "yolo_dataset_aug"
    OUT_IMG_DIR = OUT_ROOT / "images" / "train"
    OUT_LBL_DIR = OUT_ROOT / "labels" / "train"
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # 전체 Train 증강 X -> split된 train만 증강
    SPLIT_TRAIN_DIR = Path(DATA_ROOT).parent / "yolo_dataset" / "images" / "train"

    aug_tf = build_general_augmentation()

    # 전체 raw 폴더가 아니라, split된 폴더 안의 파일 리스트
    if not SPLIT_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Train split 폴더를 찾을 수 없습니다: {SPLIT_TRAIN_DIR}"
        )

    # split된 폴더에 있는 파일 이름들만 추출
    train_filenames = {
        p.name for p in SPLIT_TRAIN_DIR.iterdir() if p.suffix.lower() in IMG_EXTS
    }

    # 원본 이미지 경로(TRAIN_IMG_DIR) 중에서 split 리스트에 포함된 것만 필터링
    img_paths = sorted(
        [p for p in TRAIN_IMG_DIR.iterdir() if p.name in train_filenames]
    )
    print(
        f"[INFO] 전체 이미지 중 Train Split에 해당하는 {len(img_paths)}개만 증강을 시작합니다."
    )

    gt_cache_path = CACHE_DIR / "targets_by_filename.json"
    if not gt_cache_path.exists():
        raise FileNotFoundError(
            f"[ERROR] cache not found: {gt_cache_path}\n"
            "먼저 dataloader/mapping.py를 실행해 cache를 생성하세요."
        )

    with open(gt_cache_path, "r", encoding="utf-8") as f:
        targets_by_filename = json.load(f)

    # labels가 문자열로 들어올 수 있어서 int로 통일 (안전장치)
    targets_by_filename = defaultdict(
        lambda: {"boxes": [], "labels": []},
        targets_by_filename,
    )
    for v in targets_by_filename.values():
        v["labels"] = [int(x) for x in v.get("labels", [])]

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

            out_img = OUT_IMG_DIR / f"{stem}_aug{k + 1}{ext}"
            out_lbl = OUT_LBL_DIR / f"{stem}_aug{k + 1}.txt"

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
