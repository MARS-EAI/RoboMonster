import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner to generate data.")
    parser.add_argument('config', type=str, help="Task config file to use")
    parser.add_argument('--control-mode', type=str, default='pd_joint_pos', help="Choose control mode, [pd_joint_pos, pd_ee_pose] are supported")
    parser.add_argument('--robot', type=str, default='panda', help="Choose robot, [panda, xarm6_robotiq] are supported")
    parser.add_argument('num', type=int, help="Number of trajectories to generate.")
    parser.add_argument('variant', type=str, default='ours', help="Choose gripper-only or heterogeneous agents")
    parser.add_argument('--save-video', action='store_true', help="Save video of the generated trajectories.")
    args = parser.parse_args()

    command = (
        f"python -m planner.run "
        f"-c \"{args.config}\" " 
        f"--control-mode=\"{args.control_mode}\" "
        f"--robot=\"{args.robot}\" "
        f"-o=\"rgb\" "
        f"--render-mode=\"sensors\" "
        f"-b=\"cpu\" "
        f"-n {args.num} "
        f"-v \"{args.variant}\" "
        f"--only-count-success "
        + (f"--save-video" if args.save_video else "")
    )
    print("command: ", command)
    os.system(command)

if __name__ == "__main__":
    main()
