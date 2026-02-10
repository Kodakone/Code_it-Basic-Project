import albumentations as A
import inspect

def build_general_augmentation():
    # bbox 파라미터 (버전 호환)
    bp_kwargs = dict(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=1,
        min_visibility=0.0,
    )
    sig = inspect.signature(A.BboxParams).parameters
    if "clip" in sig:
        bp_kwargs["clip"] = True
    if "filter_lost_elements" in sig:
        bp_kwargs["filter_lost_elements"] = True
    if "check_each_transform" in sig:
        bp_kwargs["check_each_transform"] = False

    bbox_params = A.BboxParams(**bp_kwargs)

    aug_tf = A.Compose(
        [
            A.Affine(scale=(0.90, 1.10), rotate=(-10, 10), translate_percent=None, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.7),
            A.GaussianBlur(blur_limit=(3, 9), p=0.3),
        ],
        bbox_params=bbox_params,
    )
    return aug_tf
