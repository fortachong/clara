
from multiprocessing import Process, Queue, Value
import time

def stream_engine(queue, flag):
    print("Send Stream...")
    try:
        while True:
            if flag.value:
                message = {
                    'DEPTH': 'data..'
                }
                queue.put(message)
                time.sleep(1)
    except KeyboardInterrupt:
        message = {
            'STOP': 1
        }
        queue.put(message)

def gui_processor(queue, flag):
    while True:
        if flag.value:
            message = queue.get()
            print("GUI processor: ")
            print(message)


def synth_processor(queue, flag):
    while True:
        if flag.value:
            message = queue.get()
            print("Synth processor: ")
            print(message)

def process_(kind, queue, flag):
    if kind == 'main':
        stream_engine(queue, flag)
    elif kind == 'synth':
        synth_processor(queue, flag)
    else:
        gui_processor(queue, flag)


if __name__ == "__main__": 
    START = Value('b', False)
    # Create queue
    main_queue = Queue()
    synth_queue = Queue()
    points_queue = Queue()

    # Create processors
    procs = []
    kind = 'main'
    p_there = Process(target=process_, args=(kind, main_queue, START))
    p_there.start()
    kind = 'synth'
    p_synth = Process(target=process_, args=(kind, synth_queue, START))
    p_synth.start()
    kind = 'gui'
    p_gui = Process(target=process_, args=(kind, points_queue, START))
    p_gui.start()
    procs.append(p_there)
    procs.append(p_synth)
    procs.append(p_gui)
    # Start queue filling
    START.value = True
    while True:
        message = main_queue.get()
        print("Processing message...")
        print("Send to Synth...")
        print("Send to GUI")
        synth_queue.put(message)
        points_queue.put(message)

        if 'STOP' in message:
            break

    # complete the processes
    for proc in procs:
        proc.join()   
    
    
    