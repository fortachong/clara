import subprocess
from datetime import datetime
from team import banner

# Allows to configure ROIs and positions
# Use aruco markers

menu = """
** Ether theremin tools **

Show Configuration: [c]
Configuration Step 1 - Body Position: [1]
Configuration Step 2 - Antenna Position: [2]
Configuration Step 3 - ROIs: [3]
Data capture: [4]
Run: [5]
Run with Visualization: [6]
Press [q] to quit

"""
cmds = ['1', '2', '3', '4', '5', '6', 'c']
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
            
        # Configuration Step 1: Body position
        if cmd == '1':
            subprocess.run("python datasette_position_calibration.py --mode 0")
            
        # Configuration Step 2: Antenna position
        if cmd == '2':
            subprocess.run("python datasette_position_calibration.py --mode 1")

        # Configuration Step 3: Configure ROIs
        if cmd == '3':
            subprocess.run("python datasette_depth_calibration.py --mode 0")

        # Configuration Step 3: Configure ROIs
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
