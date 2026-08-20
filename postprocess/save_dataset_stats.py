import os
import re
import argparse
import h5py
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation


# TODO:

# 1) ADD LIBS IN DOCKERFILE

# 10 no
# 5 si
# 15 no
# 20 si
# 5 no

"""
Requirements: pip install h5py pandas openpyxl
Usage example: python save_dataset_stats.py --hdf5_path ../datasets/hdf5_converted/dataset_task1.h5
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Extract dataset statistics from HDF5 and generate annotation Excel file.")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the input HDF5 dataset file.")
    return parser.parse_args()


def natsort_key(s: str):
    """Key function for natural sorting (e.g., demo_0, demo_1, demo_2, ..., demo_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def get_all_dataset_info(group, prefix=''):
    """Recursively fetch all dataset paths/keys along with their shapes inside an HDF5 group."""
    datasets = {}
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            datasets[path] = item
        elif isinstance(item, h5py.Group):
            datasets.update(get_all_dataset_info(item, prefix=path))
    return datasets


def create_annotation_excel(hdf5_path: str):

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 dataset not found at path: {hdf5_path}")

    # Derive output filename from the input HDF5 path (e.g., dataset_task1.xlsx)
    base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    excel_output_path = f"{base_name}.xlsx"

    print(f"Scanning HDF5 dataset: {hdf5_path}...")

    episodes_list = []
    formatted_keys = []
    frequency_val = ""

    # Open HDF5 following the style of calc_training_steps (accessing f['data'])
    with h5py.File(hdf5_path, "r") as f:

        # Check if 'data' group exists, otherwise fallback to root keys
        if "data" in f and isinstance(f["data"], h5py.Group):
            data_grp = f["data"]
            demos = [k for k in data_grp.keys() if k.startswith("demo_")]
        else:
            data_grp = f
            demos = [k for k in data_grp.keys() if isinstance(data_grp[k], h5py.Group)]

        # Apply natural sorting to episode keys
        demos = sorted(demos, key=natsort_key)

        for idx, demo_name in enumerate(demos):

            ep_group = data_grp[demo_name]

            # 1. N_FRAMES: Try fetching from attributes, fallback to shape of first dataset
            if "num_samples" in ep_group.attrs:
                n_frames = ep_group.attrs["num_samples"]
            else:
                datasets = get_all_dataset_info(ep_group)
                ds_items = list(datasets.values())
                n_frames = ds_items[0].shape[0] if ds_items and len(ds_items[0].shape) > 0 else None

            # 2. RECOVERY FLAG
            with_recovery = "False"
            if "with_recovery" in ep_group.attrs:
                with_recovery = str(bool(ep_group.attrs["with_recovery"][()]))

            episodes_list.append({
                "Episode": demo_name,
                "N_frames": n_frames,
                "Recovery": with_recovery
            })

            # Fetch keys and frequency ONLY from the first episode (idx == 0)
            if idx == 0:
                frequency_val = ep_group.attrs.get("fps", ep_group.attrs.get("frequency", ""))
                datasets = get_all_dataset_info(ep_group)
                for ds_name, ds_obj in datasets.items():
                    shape = ds_obj.shape
                    # Ignore the first dimension (n_frames) if multi-dimensional
                    feature_shape = shape[1:] if len(shape) > 1 else shape
                    shape_str = f"({', '.join(map(str, feature_shape))})" if feature_shape else "()"
                    formatted_keys.append(f"{ds_name} {shape_str}")

    # Compute global dataset metrics
    n_episodes = len(episodes_list)
    valid_frames = [ep["N_frames"] for ep in episodes_list]
    avg_frames = round(sum(valid_frames) / len(valid_frames), 1) if valid_frames else ""
    total_recoveries = sum(1 for ep in episodes_list if ep["Recovery"] == "True")

    # Combine episodes and keys independently line by line
    max_rows = max(len(episodes_list), len(formatted_keys))
    final_rows = []

    for i in range(max_rows):
        ep_info = episodes_list[i] if i < len(episodes_list) else {"Episode": "", "N_frames": "", "Recovery": ""}
        key_info = formatted_keys[i] if i < len(formatted_keys) else ""

        final_rows.append({
            "Episode": ep_info["Episode"],
            "N_frames": ep_info["N_frames"],
            "Recovery": ep_info["Recovery"],
            "N_recoveries": total_recoveries if i == 0 else "",
            "N_episodes": n_episodes if i == 0 else "",
            "avrg_frames / episode": avg_frames if i == 0 else "",
            "Keys / Actions names": key_info,
            "Frequency (Hz)": frequency_val if i == 0 else "",
            "Rotations convention": "",
            "Positions convention": "",
            "Action order": "",
            "Task description": ""
        })

    # Create DataFrame
    df = pd.DataFrame(final_rows)

    # Export to Excel using openpyxl engine
    with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Annotations")

    # Re-open workbook with openpyxl to apply formatting and data validation (dropdowns)
    wb = load_workbook(excel_output_path)
    ws = wb["Annotations"]

    # Define Data Validation rules for dropdown lists
    dv_rot = DataValidation(
        type="list", 
        formula1='"Euler-XYZ, Euler-ZYX, Quaternion-WXYZ, Quaternion-XYZW, Axis-Angle, Rotation-Matrix"', 
        allow_blank=True
    )
    dv_pos = DataValidation(
        type="list", 
        formula1='"Absolute, Relative (wrt last action)"', 
        allow_blank=True
    )
    dv_act = DataValidation(
        type="list", 
        formula1='"positions-rotations-gripper, gripper-positions-rotations"', 
        allow_blank=True
    )

    # Attach validations to the worksheet
    ws.add_data_validation(dv_rot)
    ws.add_data_validation(dv_pos)
    ws.add_data_validation(dv_act)

    # Apply dropdown menus ONLY to the first data row (Row 2 in Excel)
    dv_rot.add("H2")  # Column H: Rotations convention
    dv_pos.add("I2")  # Column I: Positions convention
    dv_act.add("J2")  # Column J: Action order

    # Align cells top and auto-adjust column widths
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            val_str = str(cell.value or '')
            max_len = max(max_len, len(val_str))
            cell.alignment = Alignment(vertical="top")

        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    # Save final formatted workbook
    wb.save(excel_output_path)
    print(f"Successfully generated annotation file: {excel_output_path}")


if __name__ == "__main__":
    args = parse_args()
    create_annotation_excel(hdf5_path=args.hdf5_path)