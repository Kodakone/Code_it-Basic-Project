import inspect
import albumentations as A
import cv2

def build_bbox_params():
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
    return A.BboxParams(**bp_kwargs)

def build_general_augmentation():
    bbox_params = build_bbox_params()
    aug_tf = A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.7,
            ),
            A.GaussianBlur(
                blur_limit=(3, 9),
                p=0.3,
            ),
        ],
        bbox_params=bbox_params,
    )
    return aug_tf

if __name__ == "__main__":
    print("[RUNNING]", __file__)
    tf = build_general_augmentation()
    print(tf.to_dict())
