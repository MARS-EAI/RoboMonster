import argparse
import os
from pathlib import Path
import sys

def build_cmd(dataset_path: str,
              output_path: str,
              agent_num: int,
              modality: str,
              load_num: int):
    if modality == "image":
        extractor = "script/image/extract.py"
    elif modality == "pointcloud":
        extractor = (
            "script/pointcloud/extract_single.py"
            if agent_num == 1
            else "script/pointcloud/extract.py"
        )
    else:
        raise ValueError(f"Unknown modality {modality}")

    cmd = (
        f'python {extractor} '
        f'--dataset_path="{dataset_path}" '
        f'--output_path="{output_path}" '
        f'--load_num {load_num} '
        f'--agent_num {agent_num}'
    )
    return cmd


def main():
    parser = argparse.ArgumentParser(
        prog="convert.py",
        description="convert data",
    )

    parser.add_argument(
        "h5_path",
        type=str,
        help="input h5 file path, for example: data/h5_data/input.h5",
    )

    parser.add_argument(
        "--agent-num",
        type=int,
        required=True,
        choices=[1, 2],
        help="agent number, choose in [1, 2]",
    )

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["image", "pointcloud"],
        help="data type, choose in [image, pointcloud]",
    )

    parser.add_argument(
        "--load-num",
        type=int,
        default=50,
        help="How many demostrations",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output file path",
    )

    args = parser.parse_args()

    h5_path = Path(args.h5_path).resolve()
    if not h5_path.exists():
        print(f"[ERROR] {h5_path} not exist", file=sys.stderr)
        sys.exit(1)

    dataset_path = str(h5_path)
    if args.out is not None:
        output_path = str(Path(args.out).resolve())
    else:
        output_path = str(h5_path.parent.parent / h5_path.name)

    cmd = build_cmd(
        dataset_path=dataset_path,
        output_path=output_path,
        agent_num=args.agent_num,
        modality=args.modality,
        load_num=args.load_num,
    )

    print("[RUN]", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"[ERROR] Convert failed: (exit code {ret})", file=sys.stderr)
        sys.exit(ret)

    print("[OK] Output ->", output_path)

if __name__ == "__main__":
    main()
