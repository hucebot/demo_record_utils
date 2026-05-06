import h5py
import argparse
from dataclasses import dataclass
from typing import Any

"""
RUN: python open_hdf5.py /mnt/Data/converted/lift_with_images.h5 --max-depth 6 --max-children 10
"""

@dataclass(frozen=True)
class PrintOptions:
    max_depth: int
    max_children: int


def _fmt_attr_value(v: Any, max_len: int = 200) -> str:
    try:
        if isinstance(v, bytes):
            s = v.decode("utf-8", errors="replace")
        else:
            s = repr(v)
    except Exception:
        s = f"<unreprable {type(v).__name__}>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _dataset_suffix(dset: h5py.Dataset) -> str:
    parts: list[str] = []
    try:
        parts.append(f"shape={tuple(dset.shape)}")
    except Exception:
        parts.append("shape=<err>")

    return "" if not parts else "  (" + ", ".join(parts) + ")"


def _walk(name: str, obj: Any, depth: int, opts: PrintOptions) -> None:
    indent = "  " * depth
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name}/")
        if depth >= opts.max_depth:
            return

        try:
            child_names = sorted(list(obj.keys()))
        except Exception:
            child_names = []

        if opts.max_children >= 0 and len(child_names) > opts.max_children:
            shown = child_names[: opts.max_children]
            hidden = len(child_names) - len(shown)
        else:
            shown = child_names
            hidden = 0

        for cn in shown:
            try:
                child = obj[cn]
            except Exception as e:
                print(f"{indent}  {cn}  <error opening: {e}>")
                continue
            _walk(cn, child, depth + 1, opts)
        if hidden:
            print(f"{indent}  ... ({hidden} more children)")

    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}{_dataset_suffix(obj)}")
    else:
        print(f"{indent}{name}  ({type(obj).__name__})")


def inspect_hdf5(path: str, opts: PrintOptions) -> None:
    print(f"file={path}")
    with h5py.File(path, "r") as f:
        _walk("/", f["/"], 0, opts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print HDF5 structure (groups/datasets) and dataset shapes."
    )
    p.add_argument("path", help="Path to .h5/.hdf5 file")
    p.add_argument("--max-depth", type=int, default=5, help="Max recursion depth")
    p.add_argument(
        "--max-children",
        type=int,
        default=5,
        help="Max children per group (-1 = no limit)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    opts = PrintOptions(
        max_depth=args.max_depth,
        max_children=args.max_children,
    )
    inspect_hdf5(args.path, opts)


if __name__ == "__main__":
    main()
