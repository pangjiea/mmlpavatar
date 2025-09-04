import numpy as np
import argparse
import os

def convert_snapshot(input_path, output_path):
    """
    Converts a single-frame snapshot .npz file (from net_viewer_smpl_multi)
    to a format compatible with load_thuman_pose_list.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print(f"Loading snapshot from: {input_path}")
    data = np.load(input_path)

    pose = data['pose']  # Shape (165,)
    Th = data['Th']      # Shape (3,)
    beta = data['beta']  # Shape (10,)

    # The pose vector (165,) is ordered as:
    # 1. root_orient (3)
    # 2. body_pose (63)
    # 3. jaw_pose (3)
    # 4. eye_pose (6)
    # 5. hand_pose (90) -> left (45) + right (45)

    # Extract the parts needed by load_thuman_pose_list
    global_orient = pose[0:3]
    body_pose = pose[3:66]
    # Jaw (66:69) and eyes (69:75) are skipped as per load_thuman_pose_list logic
    hand_pose = pose[75:165]
    left_hand_pose = hand_pose[0:45]
    right_hand_pose = hand_pose[45:90]

    # The target format expects arrays of frames, so we add a leading dimension.
    output_data = {
        'global_orient': np.expand_dims(global_orient, axis=0),
        'body_pose': np.expand_dims(body_pose, axis=0),
        'left_hand_pose': np.expand_dims(left_hand_pose, axis=0),
        'right_hand_pose': np.expand_dims(right_hand_pose, axis=0),
        'transl': np.expand_dims(Th, axis=0),
        'betas': np.expand_dims(beta, axis=0)
    }

    print(f"Saving converted data to: {output_path}")
    np.savez(output_path, **output_data)
    print("Conversion successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert single-frame smplx.npz to thuman-compatible format.')
    parser.add_argument('input_file', type=str, help='Path to the input smplx.npz file.')
    parser.add_argument('output_file', type=str, help='Path for the output thuman-compatible .npz file.')
    args = parser.parse_args()

    convert_snapshot(args.input_file, args.output_file)
