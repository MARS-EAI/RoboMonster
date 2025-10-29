"""
How to use:
  python read_h5.py 
  python read_h5.py --input ./input/{h5 files name}.h5 --out output --mode summary
  python read_h5.py --mode full        # output full data

Output:
  data/{h5 files name}_summary.json
  data/{h5 files name}_tree.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np

MAX_ELEMENTS_FULL = 200_000
SAMPLE_FIRST_AXIS = 5

def np_to_native(x: np.ndarray) -> Union[list, int, float, str, None]:

    if x.ndim == 0:
        return x.item()
    if x.size > MAX_ELEMENTS_FULL and x.ndim >= 1:
        take = min(SAMPLE_FIRST_AXIS, x.shape[0])
        return {
            "__sample__": True,
            "sample_shape": (take,) + x.shape[1:],
            "orig_shape": x.shape,
            "data": x[:take].tolist()
        }
    return x.tolist()

def attrs_to_native(attrs: h5py.AttributeManager) -> Dict[str, Any]:
    out = {}
    for k in attrs.keys():
        v = attrs[k]
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[k] = v.item()
            else:
                out[k] = v.tolist()
        elif isinstance(v, (bytes, bytearray)):
            out[k] = v.decode("utf-8", errors="ignore")
        else:
            out[k] = v
    return out

def build_summary_node(name: str, obj: Union[h5py.Group, h5py.Dataset], mode: str) -> Dict[str, Any]:
    if isinstance(obj, h5py.Dataset):
        node = {
            "__type": "dataset",
            "name": name.split("/")[-1],
            "path": name,
            "shape": tuple(obj.shape),
            "dtype": str(obj.dtype),
            "attrs": attrs_to_native(obj.attrs)
        }
        if mode == "full":
            try:
                data = obj[()]  # 读取全部
                if isinstance(data, np.ndarray):
                    node["data"] = np_to_native(data)
                else:
                    node["data"] = data
            except Exception as e:
                node["data_error"] = f"{type(e).__name__}: {e}"
        return node
    else:
        # Group
        node = {
            "__type": "group",
            "name": name.split("/")[-1] if name else "/",
            "path": name if name else "/",
            "attrs": attrs_to_native(obj.attrs),
            "children": {}
        }
        for key in obj.keys():
            child = obj[key]
            child_name = f"{name}/{key}" if name else f"/{key}"
            node["children"][key] = build_summary_node(child_name, child, mode)
        return node

def build_tree_text(name: str, obj: Union[h5py.Group, h5py.Dataset], prefix: str = "") -> str:
    lines = []
    if isinstance(obj, h5py.Dataset):
        lines.append(f"{prefix}- {name.split('/')[-1]}  [dataset]  shape={obj.shape}, dtype={obj.dtype}")
    else:
        header = "/" if name in ("", "/") else name.split("/")[-1]
        lines.append(f"{prefix}+ {header} [group]")
        keys = list(obj.keys())
        for i, k in enumerate(keys):
            child = obj[k]
            new_prefix = prefix + ("  " if i == len(keys) - 1 else "  ")
            lines.append(build_tree_text(f"{name}/{k}" if name else f"/{k}", child, prefix + "  "))
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input/converted_example.h5", help="h5 files path")
    parser.add_argument("--out", "-o", default="output", help="output files path")
    parser.add_argument("--mode", choices=["summary", "full"], default="full",
                        help="summary: only the structure of h5 file, full: include all data in h5 file")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = in_path.stem
    json_path = out_dir / f"{base}_summary.json"
    tree_path = out_dir / f"{base}_tree.txt"

    if not in_path.exists():
        raise FileNotFoundError(f"h5 files path not exist!: {in_path}")

    with h5py.File(in_path, "r") as f:
        root_name = ""
        root_node = build_summary_node(root_name, f, args.mode)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(root_node, jf, ensure_ascii=False, indent=2)

        tree_text = build_tree_text("", f)
        with open(tree_path, "w", encoding="utf-8") as tf:
            tf.write(tree_text + "\n")

    print(f"finish: {json_path}")
    print(f"finish: {tree_path}")

if __name__ == "__main__":
    main()
