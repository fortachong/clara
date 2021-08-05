
from multiprocessing import Process, Queue, Value
import time

def stream_engine(queue, flag_start, flag_stop):
    print("Send Stream...")
    try:
        while True:
            if flag_start.value:
                message = {
                    'DEPTH': 'data..'
                }
                queue.put(message)
                time.sleep(1)
            if flag_stop.value:
                print("Stopping depth streaming...")
                break

    except KeyboardInterrupt:
        flag_stop.value = True

def gui_processor(queue, flag_start, flag_stop):
    try:
        while True:
            if flag_start.value:
                message = queue.get()
                print("GUI processor: ")
                print(message)
            if flag_stop.value:
                print("Stopping gui...")
                break                
    except KeyboardInterrupt:
        flag_stop.value = True            


def synth_processor(queue, flag_start, flag_stop):
    try:
        while True:
            if flag_start.value:
                message = queue.get()
                print("Synth processor: ")
                print(message)
            if flag_stop.value:
                print("Stopping synth...")
                break                
    except KeyboardInterrupt:
        flag_stop.value = True

def process_(kind, queue, flag_start, flag_stop):
    if kind == 'main':
        stream_engine(queue, flag_start, flag_stop)
    elif kind == 'synth':
        synth_processor(queue, flag_start, flag_stop)
    else:
        gui_processor(queue, flag_start, flag_stop)


if __name__ == "__main__": 
    START = Value('b', False)
    STOP = Value('b', False)
    # Create queue
    main_queue = Queue()
    synth_queue = Queue()
    points_queue = Queue()

    # Create processors
    procs = []
    kind = 'main'
    p_there = Process(target=process_, args=(kind, main_queue, START, STOP))
    p_there.start()
    kind = 'synth'
    # p_synth = Process(target=process_, args=(kind, synth_queue, START, STOP))
    p_synth = Process(target=synth_processor, args=(synth_queue, START, STOP))
    p_synth.start()
    kind = 'gui'
    p_gui = Process(target=process_, args=(kind, points_queue, START, STOP))
    p_gui.start()
    procs.append(p_there)
    procs.append(p_synth)
    procs.append(p_gui)
    # Start queue filling
    START.value = True
    try:
        while True:
            message = main_queue.get()
            print("Processing message...")
            print("Send to Synth...")
            print("Send to GUI")
            synth_queue.put(message)
            points_queue.put(message)
            if STOP.value:
                print("Stopping coordinator...")
                break               
    except KeyboardInterrupt:
        STOP.value = True
 
    # complete the processes
    for proc in procs:
        proc.join()   
    
    
    