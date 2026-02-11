"""
sanity_check.py
- 프로젝트 경로/환경변수/데이터셋/모델 가중치/매핑 파일을 한 번에 점검하는 스크립트

권장 위치:
  <project_root>/bin/sanity_check.py

실행:
  cd <project_root>
  python bin/sanity_check.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

from dotenv import load_dotenv

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None

# -----------------------------
# Utils
# -----------------------------
@dataclass
class CheckResult:
    ok: bool
    title: str
    detail: str
    hint: str = ""


def find_project_root(start: Path) -> Path:
    """
    프로젝트 루트를 위로 올라가며 탐색.
    기준(우선순위):
      1) .env 존재
      2) data 폴더 존재
      3) runs 폴더 존재
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / ".env").exists():
            return cur
        if (cur / "data").exists() and (cur / "runs").exists():
            return cur
        cur = cur.parent
    # fallback: start의 부모
    return start.resolve().parent


def read_int_lines(path: Path) -> List[int]:
    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # 숫자만 남기기 (예: "123  # comment" 같은 경우)
            s = s.split("#", 1)[0].strip()
            if not s:
                continue
            try:
                ids.append(int(s))
            except ValueError:
                # 숫자로 변환 불가한 라인은 무시
                continue
    return ids


def fmt_bool(ok: bool) -> str:
    return "✅" if ok else "❌"


def print_block(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# -----------------------------
# Checks
# -----------------------------
def check_env(project_root: Path) -> List[CheckResult]:
    env_path = project_root / ".env"
    ok = env_path.exists()
    res: List[CheckResult] = []
    res.append(
        CheckResult(
            ok=ok,
            title=".env 존재 여부",
            detail=f"{env_path}",
            hint="프로젝트 루트에 .env가 있어야 YOLO_PATH, TARGET_MODEL_NAME 등을 읽을 수 있어요.",
        )
    )

    # .env 로드(있든 없든 실행은 하되, 없으면 이후가 다 fail로 찍힘)
    load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=True)

    yolo_path = os.getenv("YOLO_PATH", "")
    target_name = os.getenv("TARGET_MODEL_NAME", "")

    res.append(
        CheckResult(
            ok=bool(yolo_path),
            title="YOLO_PATH 설정",
            detail=f"YOLO_PATH={yolo_path!r}",
            hint="예) YOLO_PATH=D:/VS Test/Code_it-Basic-Project/data/yolo_dataset_aug",
        )
    )
    res.append(
        CheckResult(
            ok=bool(target_name),
            title="TARGET_MODEL_NAME 설정",
            detail=f"TARGET_MODEL_NAME={target_name!r}",
            hint="train.py가 학습 후 .env의 TARGET_MODEL_NAME을 자동 업데이트합니다.",
        )
    )

    # YOLO_PATH 실경로 체크
    if yolo_path:
        p = Path(yolo_path)
        res.append(
            CheckResult(
                ok=p.exists(),
                title="YOLO_PATH 경로 존재",
                detail=str(p),
                hint="해당 폴더가 실제로 존재해야 dataset.yaml, labels/train 등이 정상입니다.",
            )
        )
    return res


def check_dataset_yaml(project_root: Path, yolo_path: Path) -> List[CheckResult]:
    res: List[CheckResult] = []
    yaml_path = yolo_path / "dataset.yaml"
    if not yaml_path.exists():
        res.append(
            CheckResult(
                ok=False,
                title="dataset.yaml 존재 여부",
                detail=str(yaml_path),
                hint="YOLO_PATH는 dataset.yaml이 있는 데이터셋 루트를 가리켜야 합니다.",
            )
        )
        return res

    res.append(CheckResult(True, "dataset.yaml 존재 여부", str(yaml_path)))

    if yaml is None:
        res.append(
            CheckResult(
                ok=False,
                title="PyYAML 설치 여부",
                detail="import yaml 실패",
                hint="pip install pyyaml 로 설치 후 다시 실행하세요.",
            )
        )
        return res

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        res.append(
            CheckResult(
                ok=False,
                title="dataset.yaml 파싱",
                detail=f"파싱 실패: {e}",
                hint="dataset.yaml이 깨졌거나 인코딩 문제가 있을 수 있어요.",
            )
        )
        return res

    nc = data.get("nc", None)
    names = data.get("names", None)
    train = data.get("train", None)
    val = data.get("val", None)

    res.append(CheckResult(isinstance(nc, int), "nc(클래스 수) 확인", f"nc={nc!r}", "nc는 정수여야 합니다."))
    if isinstance(names, list):
        res.append(CheckResult(True, "names 리스트 확인", f"len(names)={len(names)}"))
    else:
        res.append(CheckResult(False, "names 리스트 확인", f"names={type(names).__name__}", "names가 리스트인지 확인하세요."))

    # nc vs names 길이
    if isinstance(nc, int) and isinstance(names, list):
        res.append(
            CheckResult(
                ok=(len(names) == nc),
                title="nc == len(names) 일치",
                detail=f"nc={nc}, len(names)={len(names)}",
                hint="이 값이 어긋나면 학습/추론에서 클래스 매핑이 꼬일 수 있어요.",
            )
        )

    # train/val 경로 존재 확인(문자열이면 상대/절대 모두 가능)
    def _path_ok(pv) -> Tuple[bool, str]:
        if not pv:
            return False, "없음"
        try:
            p = Path(str(pv))
            if not p.is_absolute():
                p = (project_root / p).resolve()
            return p.exists(), str(p)
        except Exception:
            return False, f"해석 불가: {pv!r}"

    ok_train, train_path = _path_ok(train)
    ok_val, val_path = _path_ok(val)
    res.append(CheckResult(ok_train, "train 경로 존재", train_path, "dataset.yaml의 train 경로를 확인하세요."))
    res.append(CheckResult(ok_val, "val 경로 존재", val_path, "dataset.yaml의 val 경로를 확인하세요."))

    # labels/images 폴더 구조 확인
    res.append(CheckResult((yolo_path / "images" / "train").exists(), "images/train 존재", str(yolo_path / "images" / "train")))
    res.append(CheckResult((yolo_path / "labels" / "train").exists(), "labels/train 존재", str(yolo_path / "labels" / "train")))

    return res


def check_labels(yolo_path: Path) -> List[CheckResult]:
    res: List[CheckResult] = []
    lbl_dir = yolo_path / "labels" / "train"
    if not lbl_dir.exists():
        res.append(CheckResult(False, "labels/train 라벨 폴더", str(lbl_dir), "YOLO_PATH가 올바른지 확인하세요."))
        return res

    txts = list(lbl_dir.glob("*.txt"))
    empties = 0
    bad_lines = 0
    max_cls = -1

    for t in txts[:2000]:  # 너무 많으면 느릴 수 있어서 상한
        try:
            lines = t.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        if not any(line.strip() for line in lines):
            empties += 1
            continue
        for line in lines:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                bad_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                max_cls = max(max_cls, cls)
            except Exception:
                bad_lines += 1

    res.append(CheckResult(True, "라벨 파일 개수", f"{len(txts)} files in {lbl_dir}"))
    res.append(
        CheckResult(
            ok=(empties == 0),
            title="빈 라벨 파일(상위 2000개 검사)",
            detail=f"empty={empties}",
            hint="빈 라벨이 많으면 split/증강 과정에서 라벨 생성이 깨졌을 가능성이 있어요.",
        )
    )
    res.append(
        CheckResult(
            ok=(bad_lines == 0),
            title="라벨 형식 오류(상위 2000개 검사)",
            detail=f"bad_lines={bad_lines}",
            hint="YOLO 라벨은 보통: cls x y w h (정규화) 형식입니다.",
        )
    )
    if max_cls >= 0:
        res.append(CheckResult(True, "라벨 내 최대 cls(상위 2000개 검사)", f"max_cls={max_cls}"))
    return res


def check_real_class_ids(project_root: Path, expected_nc: Optional[int]) -> List[CheckResult]:
    res: List[CheckResult] = []
    p = project_root / "bin" / "real_class_ids.txt"
    if not p.exists():
        res.append(
            CheckResult(
                ok=False,
                title="real_class_ids.txt 존재 여부",
                detail=str(p),
                hint="extraction.py는 <project_root>/bin/real_class_ids.txt를 읽습니다.",
            )
        )
        return res

    ids = read_int_lines(p)
    res.append(CheckResult(True, "real_class_ids.txt 로드", f"{p} (len={len(ids)})"))
    if expected_nc is not None:
        res.append(
            CheckResult(
                ok=(len(ids) == expected_nc),
                title="len(real_class_ids) == nc",
                detail=f"len={len(ids)}, nc={expected_nc}",
                hint="길이가 다르면 제출 category_id 매핑이 틀어집니다.",
            )
        )
    return res


def check_models(project_root: Path, target_name: str) -> List[CheckResult]:
    res: List[CheckResult] = []
    if not target_name:
        res.append(CheckResult(False, "모델 체크", "TARGET_MODEL_NAME이 비어있음", "train.py를 먼저 실행해 .env를 업데이트하세요."))
        return res

    ft = project_root / "runs" / "fine_tuning" / f"{target_name}_FT" / "weights" / "best.pt"
    tr = project_root / "runs" / "train" / target_name / "weights" / "best.pt"

    res.append(CheckResult(ft.exists(), "Fine-tuning best.pt", str(ft), "fine_tuning.py 실행 여부/경로 확인"))
    res.append(CheckResult(tr.exists(), "Train best.pt", str(tr), "train.py 실행 여부/경로 확인"))

    # 실제 추론에서 어떤 모델이 선택될지 요약
    if ft.exists():
        chosen = ft
        why = "FT 모델이 존재하므로 test.py/extraction.py가 FT를 우선 사용"
    elif tr.exists():
        chosen = tr
        why = "FT가 없으므로 train best.pt 사용"
    else:
        chosen = None
        why = "best.pt를 찾지 못함"

    res.append(CheckResult(bool(chosen), "추론에 사용될 모델", str(chosen) if chosen else "없음", why))
    return res


def check_test_images(project_root: Path) -> List[CheckResult]:
    res: List[CheckResult] = []
    d = project_root / "data" / "raw" / "test_images"
    ok = d.exists()
    res.append(CheckResult(ok, "test_images 폴더", str(d), "test.py/extraction.py가 이 경로를 사용합니다."))
    if ok:
        pngs = list(d.glob("*.png"))
        res.append(CheckResult(len(pngs) > 0, "test_images PNG 개수", f"{len(pngs)} files"))
    return res


def check_ultralytics_and_gpu() -> List[CheckResult]:
    res: List[CheckResult] = []
    # ultralytics 설치 여부
    try:
        import ultralytics  # type: ignore

        res.append(CheckResult(True, "ultralytics 설치", f"version={getattr(ultralytics, '__version__', 'unknown')}"))
    except Exception as e:
        res.append(CheckResult(False, "ultralytics 설치", f"import 실패: {e}", "pip install ultralytics"))
        return res

    # torch / cuda
    try:
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        res.append(CheckResult(True, "torch 설치", f"torch={getattr(torch, '__version__', 'unknown')}"))
        res.append(CheckResult(cuda, "CUDA 사용 가능", f"cuda_available={cuda}", "train.py에서 device=0 사용. GPU 없으면 device='cpu'로 변경 필요"))
        if cuda:
            res.append(CheckResult(True, "CUDA device count", f"count={torch.cuda.device_count()}"))
    except Exception as e:
        res.append(CheckResult(False, "torch 설치", f"import 실패: {e}", "GPU 학습을 안 하면 torch는 없어도 되지만, ultralytics 내부에서 사용합니다."))
    return res


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    print_block("📌 Project Root")
    print(f"{project_root}")

    # 1) env
    print_block("1) 환경 변수(.env) 점검")
    env_results = check_env(project_root)
    for r in env_results:
        print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
        if (not r.ok) and r.hint:
            print(f"  ↳ 힌트: {r.hint}")

    # read env values after loading
    yolo_path_str = os.getenv("YOLO_PATH", "")
    target_name = os.getenv("TARGET_MODEL_NAME", "")

    # 2) ultralytics & gpu
    print_block("2) 라이브러리/하드웨어 점검")
    for r in check_ultralytics_and_gpu():
        print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
        if (not r.ok) and r.hint:
            print(f"  ↳ 힌트: {r.hint}")

    # 3) dataset yaml
    expected_nc: Optional[int] = None
    if yolo_path_str:
        yolo_path = Path(yolo_path_str)
        print_block("3) 데이터셋(dataset.yaml) 점검")
        ds_results = check_dataset_yaml(project_root, yolo_path)
        for r in ds_results:
            print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
            if (not r.ok) and r.hint:
                print(f"  ↳ 힌트: {r.hint}")

        # parse nc again for downstream checks
        yaml_path = yolo_path / "dataset.yaml"
        if yaml_path.exists() and yaml is not None:
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                if isinstance(data.get("nc", None), int):
                    expected_nc = int(data["nc"])
            except Exception:
                expected_nc = None

        print_block("4) 라벨(labels/train) 품질 점검")
        for r in check_labels(yolo_path):
            print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
            if (not r.ok) and r.hint:
                print(f"  ↳ 힌트: {r.hint}")
    else:
        print_block("3) 데이터셋 점검 스킵")
        print("YOLO_PATH가 비어 있어서 dataset.yaml/labels 점검을 건너뜁니다.")

    # 5) real_class_ids
    print_block("5) 제출 매핑(real_class_ids.txt) 점검")
    for r in check_real_class_ids(project_root, expected_nc):
        print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
        if (not r.ok) and r.hint:
            print(f"  ↳ 힌트: {r.hint}")

    # 6) 모델 가중치
    print_block("6) 모델 가중치(best.pt) 점검")
    for r in check_models(project_root, target_name):
        print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
        if (not r.ok) and r.hint:
            print(f"  ↳ 힌트: {r.hint}")

    # 7) test_images
    print_block("7) 테스트 이미지(test_images) 점검")
    for r in check_test_images(project_root):
        print(f"{fmt_bool(r.ok)} {r.title}\n  - {r.detail}")
        if (not r.ok) and r.hint:
            print(f"  ↳ 힌트: {r.hint}")

    # Summary
    print_block("✅ 요약")
    all_results = []
    all_results += env_results

    # Quick exit code: if any critical checks fail
    critical_fail = any(
        (not r.ok) and r.title in [
            ".env 존재 여부",
            "YOLO_PATH 설정",
            "YOLO_PATH 경로 존재",
        ]
        for r in env_results
    )

    if critical_fail:
        print("❌ 필수 항목이 실패했습니다. 위 힌트를 참고해서 경로/환경변수를 먼저 수정하세요.")
        return 2

    print("완료! 위에서 ❌로 표시된 항목만 순서대로 해결하면 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
