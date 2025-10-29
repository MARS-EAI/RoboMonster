import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner with task config input")
    parser.add_argument('config', type=str, help="Task config file to use")
    parser.add_argument('--control-mode', type=str, default='pd_joint_pos', help="Choose control mode, [pd_joint_pos, pd_ee_pose] are supported")
    parser.add_argument('--robot', type=str, default='panda', help="Choose robot, [panda, xarm6_robotiq] are supported")
    parser.add_argument('variant', type=str, default='ours', help="Choose gripper-only or heterogeneous agents")
    args = parser.parse_args()

    command = (
        f"python -m planner.run "
        f"-c \"{args.config}\" "
        f"--control-mode=\"{args.control_mode}\" "
        f"--robot=\"{args.robot}\" "
        f"--render-mode=\"human\" "
        f"-b=\"cpu\" "
        f"-n 1 "
        f"-v \"{args.variant}\" "
        f"--vis"
    )

    os.system(command)

if __name__ == "__main__":
    main()
