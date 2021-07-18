import tkinter as tk
import os
from team import banner

# To be done...
# 1. Calibration - Position
def cal_step_1():
    pass
    
# 2. Hand Movement Range
def cal_step_2():
    # cmd = "python datasette_depth_calibration.py --mode 0"
    pass

if __name__ == "__main__":
    print(banner)
    # GUI 
    window = tk.Tk()
    window.title("Ether v0.2.3 GUI")

    # Calibration Step 1
    btn1 = tk.Button(
        text="1. Calibration: Position",
        font='sans 16 bold',
        width=40,
        height=4,
        bg="#282928",
        fg="#C6DDF0",
        command=cal_step_1
    )
    btn1.pack()

    # Calibration Step 2
    btn2 = tk.Button(
        text="2. Calibration: Hand Range",
        font='sans 16 bold',
        width=40,
        height=4,
        bg="#282928",
        fg="#C6DDF0",
        command=cal_step_2
    )
    btn2.pack()
    window.mainloop()