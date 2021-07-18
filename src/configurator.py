import subprocess
from datetime import datetime
from team import banner

# Allows to configure ROIs and positions
# Use aruco markers

menu = """
Welcome to Ether theremin tools
Configuration Step 1 - Body Position: Press [1]
Configuration Step 2 - Antenna Position: Press [2]
Configuration Step 3 - ROIs: Press [3]
Data capture: Press [4]
Press [q] to quit

"""
cmds = ['1', '2', '3', '4']
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


        