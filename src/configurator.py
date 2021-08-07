import subprocess
from datetime import datetime
from team import banner

# Allows to configure ROIs and positions
# Use aruco markers

menu = """
** Ether theremin ROIs configuration and capture tools **

Devices Info: [i]
Show Configuration: [c]
Configuration Step 1 - Body Position: [1]
Configuration Step 2 - Antenna Position: [2]
Configuration Step 3 - ROIs: [3]
Data capture (30 fps): [4]
Run (30 fps): [5]
Run with Visualization: [6]
Data capture (Hand Positions): [h]
Press [q] to quit

"""
hand_menu = """"
Valid positions: [0], [1], [2], or [3]
Press [q] to return to menu

"""

cmds = ['1', '2', '3', '4', '5', '6', 'c', 'i', 'h']
positions = ['0', '1', '2', '3']
if __name__ == "__main__":
    print(banner)
    while True:
        print(menu)
        cmd = input("Enter command: ")
        if cmd == 'q':
            break
        
        if cmd not in cmds:
            print("Command not recognized")

        # Configuration Step 1: Body position
        if cmd == 'c':
            # Show Configuration:
            # Intrinsics (Right Matrix)
            # ROI files (body, antenna, hands)
            # Conversions
            # Depth min and max
            # Map (matplotlib)
            subprocess.run("python show_config.py")

        if cmd == 'i':
            # Get info
            subprocess.run("python dinfo.py")
            
        # Configuration Step 1: Body position
        if cmd == '1':
            subprocess.run("python datasette_position_calibration.py --mode 0")
            
        # Configuration Step 2: Antenna position
        if cmd == '2':
            subprocess.run("python datasette_position_calibration.py --mode 1")

        # Configuration Step 3: Configure ROIs
        if cmd == '3':
            subprocess.run("python datasette_depth_calibration.py --mode 0")

        # Capture data
        if cmd == '4':
            dt = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
            prefix = f"c_{dt}"
            subprocess.run(f"python datasette_depth_calibration.py --mode 1 --prefix {prefix}")

        # Run
        if cmd == '5':
            subprocess.run("python roi_depth_visualizer.py --plot=0")

        # Visualization
        if cmd == '6':
            subprocess.run("python roi_depth_visualizer.py --plot=1")

        # Data capture
        if cmd == 'h':
            # Loop to capture data
            
            while True:
                print(hand_menu)
                pos = input("Select position to capture: ")
                if pos == 'q':
                    break

                if pos not in positions:
                    print("Invalid position")
                else:
                    label = 'P' + pos
                    subprocess.run(f"python datasette_hand_recorder.py --fps 5 --label {label}")
