import os
import re
import argparse
import h5py
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation

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

    base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    excel_output_path = f"{base_name}.xlsx"

    print(f"Scanning HDF5 dataset: {hdf5_path}...")

    episodes_list = []
    formatted_keys = []
    frequency_val = 30.0

    # Open HDF5 file
    with h5py.File(hdf5_path, "r") as f:

        if "data" in f and isinstance(f["data"], h5py.Group):
            data_grp = f["data"]
            demos = [k for k in data_grp.keys() if k.startswith("demo_")]
        else:
            data_grp = f
            demos = [k for k in data_grp.keys() if isinstance(data_grp[k], h5py.Group)]

        demos = sorted(demos, key=natsort_key)

        for idx, demo_name in enumerate(demos):

            ep_group = data_grp[demo_name]

            # 1. Fetch N_FRAMES per episode
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
                raw_freq = ep_group.attrs.get("fps", ep_group.attrs.get("frequency", None))
                if raw_freq is not None and float(raw_freq) > 0:
                    frequency_val = float(raw_freq)

                datasets = get_all_dataset_info(ep_group)
                for ds_name, ds_obj in datasets.items():
                    shape = ds_obj.shape
                    feature_shape = shape[1:] if len(shape) > 1 else shape
                    shape_str = f"({', '.join(map(str, feature_shape))})" if feature_shape else "()"
                    formatted_keys.append(f"{ds_name} {shape_str}")

    # Compute global dataset metrics
    n_episodes = len(episodes_list)
    valid_frames = [ep["N_frames"] for ep in episodes_list if ep["N_frames"] is not None]
    n_total_frames = sum(valid_frames) if valid_frames else 0
    avg_frames = round(n_total_frames / len(valid_frames), 1) if valid_frames else ""
    total_recoveries = sum(1 for ep in episodes_list if ep["Recovery"] == "True")

    # Dynamic Excel formula referencing cell F2 (N_total_frames) and cell I2 (Frequency)
    # Formula: =ROUND(F2 / (I2 * 60), 2)
    duration_excel_formula = "=ROUND(F2/(I2*60), 2)" if n_total_frames > 0 else 0.0

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
            "N_total_frames": n_total_frames if i == 0 else "",
            "avrg_frames / episode": avg_frames if i == 0 else "",
            "Keys / Actions names": key_info,
            "Frequency (Hz)": frequency_val if i == 0 else "",
            "Dataset duration [minutes]": duration_excel_formula if i == 0 else "",
            "Rotations convention": "Quaternion-XYZW" if i == 0 else "",
            "Positions convention": "Absolute" if i == 0 else "",
            "Action order": "x,y,z,qx,qy,qz,qw,gripper" if i == 0 else "",
            "Task description": "" if i == 0 else "",
            "Git commit demo_record_utils (branch: add_recovery_button)": "576fe60376ef3bebed262ff44c2d644f4a21abb5" if i == 0 else "",
            "Git commit multipanda_ros2 (branch: dionisis-wip)": "cef5b0d6e49204aea20ce7dbbe53d3888a9a727c" if i == 0 else "",
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

    # Attach validations to the worksheet
    ws.add_data_validation(dv_rot)
    ws.add_data_validation(dv_pos)

    # Apply dropdown menus ONLY to the first data row (Row 2 in Excel)
    dv_rot.add("K2")  # Column K: Rotations convention
    dv_pos.add("L2")  # Column L: Positions convention

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