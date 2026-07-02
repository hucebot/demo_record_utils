import rerun as rr
import h5py
import argparse
import os
import subprocess

"""
Run: python vis/rerun_viewer.py --dataset datasets/cubes.h5
"""

def visualize_hdf5(file_path, output_path, blueprint_path=None, target_demo=None):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Initializing Rerun (saving to {output_path})...")
    rr.init("ForceVAM_Viewer")

    if blueprint_path and os.path.exists(blueprint_path):
        print(f"Applying saved blueprint blueprint: {blueprint_path}")
        rr.log_file_from_path(blueprint_path)

    with h5py.File(file_path, 'r') as f:
        if 'data' not in f:
            print("Error: No 'data' group found in HDF5.")
            return

        data_group = f['data']
        demos_to_load = [target_demo] if target_demo else list(data_group.keys())

        for demo_name in demos_to_load:
            if demo_name not in data_group:
                print(f"Warning: '{demo_name}' not found. Skipping.")
                continue

            demo = data_group[demo_name]

            # 1. Dynamically discover all arrays
            dataset_paths = []
            def collect_datasets(name, node):
                if isinstance(node, h5py.Dataset):
                    dataset_paths.append(name)

            demo.visititems(collect_datasets)

            # Determine number of samples
            num_samples = demo.attrs.get('num_samples', 0)
            if num_samples == 0 and 'actions' in demo:
                num_samples = len(demo['actions'])

            print(f"Loading {demo_name} ({num_samples} frames)...")

            # Match position and quaternion keys for 3D Transforms (Robomimic format)
            tf_bases = {}
            for path in dataset_paths:
                if path.endswith('_pos'):
                    base = path[:-4]
                    if f"{base}_quat" in dataset_paths:
                        tf_bases[base] = (path, f"{base}_quat")

            skip_paths = set()
            for p_pos, p_quat in tf_bases.values():
                skip_paths.add(p_pos)
                skip_paths.add(p_quat)

            # 2. Iterate through time and log data
            for i in range(num_samples):
                rr.set_time("frame", sequence=i)

                for path in dataset_paths:
                    val = demo[path][i]

                    # --- CAMERAS (Top Row) ---
                    if 'cam' in path.lower() or 'image' in path.lower():
                        small_img = val[::2, ::2]
                        # Groups all cameras under a "Cameras" space
                        rr.log(f"Cameras/{path.split('/')[-1]}", rr.Image(small_img))
                        continue

                    # --- KINEMATICS (2nd Row) ---
                    if 'curr_ee_pos' in path:
                        # Puts X, Y, Z on the SAME plot
                        rr.log("Kinematics/EE_Position/X", rr.Scalars(val[0]))
                        rr.log("Kinematics/EE_Position/Y", rr.Scalars(val[1]))
                        rr.log("Kinematics/EE_Position/Z", rr.Scalars(val[2]))

                        # Keeps the 3D visualizer
                        rr.log(
                            "Kinematics/3D_Transform",
                            rr.Transform3D(
                                translation=val[0:3],
                                rotation=rr.Quaternion(xyzw=val[3:7])
                            )
                        )
                        continue

                    if 'gripper_width' in path:
                        rr.log("Kinematics/Gripper_Width", rr.Scalars(val.item()))
                        continue

                    # --- DYNAMICS (3rd Row) ---
                    if 'ee_force' in path:
                        rr.log("Dynamics/EE_Force/X", rr.Scalars(val[0]))
                        rr.log("Dynamics/EE_Force/Y", rr.Scalars(val[1]))
                        rr.log("Dynamics/EE_Force/Z", rr.Scalars(val[2]))
                        continue

                    if 'ee_torque' in path:
                        rr.log("Dynamics/EE_Torque/X", rr.Scalars(val[0]))
                        rr.log("Dynamics/EE_Torque/Y", rr.Scalars(val[1]))
                        rr.log("Dynamics/EE_Torque/Z", rr.Scalars(val[2]))
                        continue

                    # --- JOINTS (Grouped 7-line plots) ---
                    if 'joint_pos' in path:
                        for j in range(7): rr.log(f"Joints/Positions/j{j}", rr.Scalars(val[j]))
                        continue
                    if 'joint_vel' in path:
                        for j in range(7): rr.log(f"Joints/Velocities/j{j}", rr.Scalars(val[j]))
                        continue
                    if 'joint_torques' in path:
                        for j in range(7): rr.log(f"Joints/Torques/j{j}", rr.Scalars(val[j]))
                        continue

                    # --- ACTIONS ---
                    if path == 'actions':
                        for j in range(val.size): rr.log(f"Actions/dim_{j}", rr.Scalars(val[j]))
                        continue

                    # Hack to get transformation out of 3d _pos
                    if '_pos' in path and getattr(val, 'size', 0) == 3:
                        # 1. Log it as a 3D Transform so it appears in the 3D Space View
                        rr.log(
                            f"3D_World/{path}",
                            rr.Transform3D(translation=val)
                        )
                        # 2. Also log the scalars so you can view them on a line chart
                        rr.log(f"Kinematics/Positions/{path}/X", rr.Scalars(val[0]))
                        rr.log(f"Kinematics/Positions/{path}/Y", rr.Scalars(val[1]))
                        rr.log(f"Kinematics/Positions/{path}/Z", rr.Scalars(val[2]))
                        continue
                    # --- FALLBACK FOR ANYTHING ELSE ---
                    if val.ndim == 1:
                        if val.size == 1:
                            rr.log(f"Other/{path}", rr.Scalars(val.item()))
                        else:
                            for j, j_val in enumerate(val):
                                rr.log(f"Other/{path}/dim_{j}", rr.Scalars(j_val))
    # 3. Save to disk
    rr.save(output_path)
    print(f"\n✅ Success! Dataset compiled to {output_path}")
    print(f"🚀 Launching Web Viewer (Press Ctrl+C to exit)...")

    # 4. Automatically launch the viewer
    try:
        subprocess.run(["python3", "-m", "rerun", output_path, "--web-viewer", "--memory-limit", "16GB"])
    except KeyboardInterrupt:
        print("\nViewer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Robomimic/ForceVAM HDF5 Visualizer")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .hdf5 dataset file")
    parser.add_argument("--demo", type=str, default="demo_0", help="Specific demo to load (e.g., demo_0). Loads all if not set.")

    # Updated defaults:
    parser.add_argument("--blueprint", type=str, default="assets/rerun/blueprints/franka_blueprint.rbl", help="Path to saved Rerun blueprint (.rbl)")
    parser.add_argument("--output", "-o", type=str, default="assets/rerun/converted_datasets/recording.rrd", help="Output .rrd file path")

    args = parser.parse_args()

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    visualize_hdf5(args.dataset, args.output, args.blueprint, args.demo)