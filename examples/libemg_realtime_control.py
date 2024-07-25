from libemg.data_handler import OnlineDataHandler
from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.streamers import emager_streamer
from libemg.filtering import Filter
from libemg.shared_memory_manager import SharedMemoryManager

from emager_py.utils.find_usb import virtual_port
import emager_py.torch.models as etm
import emager_py.utils.utils as eutils
from emager_py.visualization import realtime_gui
import emager_py.utils.gestures_json as gjutils
from emager_py.control.smart_hand_control import SmartHandControl

import os
import torch
import time
from multiprocessing.connection import Connection
from multiprocessing import Lock, Process, Pipe
import numpy as np
import threading


eutils.set_logging()


MODEL_PATH = "C:\GIT\Datasets/Libemg/Demo/libemg_torch_cnn_Demo_904_24-07-19_13h14.pth"
MEDIA_PATH = "C:\GIT\emager-py\media-test/"

NUM_CLASSES = 5
WINDOW_SIZE=200
WINDOW_INCREMENT=10
MAJORITY_VOTE=7

VIRTUAL = True


# PREDICTOR
def update_labels_process(gui:realtime_gui.RealTimeGestureUi, smm_items:list, stop_event:threading.Event, conn:Connection=None):
    smm = SharedMemoryManager()
    while not stop_event.is_set():
        check = True
        for item in smm_items:
            tag, shape, dtype, lock = item
            if not smm.find_variable(tag, shape, dtype, lock):
                # wait for the variable to be created
                check = False
                break
        if not check:
            continue

        # Read from shared memory
        classifier_output = smm.get_variable("classifier_output")
        
        # The most recent output is at index 0
        latest_output = classifier_output[0]
        output_data = {
            "timestamp": np.double(latest_output[0]),
            "prediction": int(latest_output[1]),
            "probability": np.double(latest_output[2]),
        }
        print(f"Sending data: ({(output_data['prediction'])}) : {output_data} ")

        gui.update_index(output_data["prediction"])

        if conn is not None:
            print(" "*80,"... sending data ...")
            conn.send(output_data)

        time.sleep(0.45)

def run_predicator(conn: Connection=None):

    # Get data port
    if VIRTUAL:
        DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
        PORT = virtual_port(DATASET_PATH)
        print("Data generator thread started")
        time.sleep(3)
    else:
        PORT = None

    # Create data handler and streamer
    p, smi = emager_streamer(specified_port=PORT)
    print(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)

    filter = Filter(1000)
    notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
    filter.install_filters(notch_filter_dictionary)
    bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
    filter.install_filters(bandpass_filter_dictionary)
    odh.install_filter(filter)
    print("Data handler created")

    # Choose feature group and classifier
    fe = FeatureExtractor()
    fg = ["MAV"]
    print("Feature group: ", fg)

    # Verify model loading and state dict compatibility
    model = etm.EmagerCNN((4, 16), NUM_CLASSES, -1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        # Handle error (e.g., exit or attempt a recovery)

    classi = EMGClassifier()
    classi.add_majority_vote(MAJORITY_VOTE)
    classi.classifier = model.eval()

    # Ensure OnlineEMGClassifier is correctly set up for data handling and inference
    smm_items=[["classifier_output", (100,3), np.double, Lock()], #timestamp, class prediction, confidence
                        ["classifier_input", (100,1+64), np.double, Lock()], # timestamp, <- features ->
                        ["adapt_flag", (1,1), np.int32, Lock()],
                        ["active_flag", (1,1), np.int8, Lock()]]
    oclassi = OnlineEMGClassifier(classi, WINDOW_SIZE, WINDOW_INCREMENT, odh, fg, std_out=False, smm=True, file=False, smm_items=smm_items)
    
    # Create GUI
    files = gjutils.get_images_list(MEDIA_PATH)
    print("Files: ", files)
    print("Creating GUI...")
    gui = realtime_gui.RealTimeGestureUi(files)
    
    
    stop_event = threading.Event()
    updateLabelTask = threading.Thread(target=update_labels_process, args=(gui, smm_items, stop_event, conn))

    try:
        print("Starting classification...")
        oclassi.run(block=False)
        print("Starting thread...")
        updateLabelTask.start()
        print("Starting GUI...")
        gui.run()
        # while True:
        #     time.sleep(1)

    except Exception as e:
        print(f"Error during classification: {e}")
    finally :
        if conn is not None:
            conn.send("exit")
        stop_event.set()
        oclassi.stop_running()
        print("Exiting")



# COMMUNICATOR

def run_communicator(conn: Connection=None):
    try:
        hand_comm = SmartHandControl(deviceName="testpico", mode="BLE")
        hand_comm.connect()

        # Main loop to read input from stdin
        print("Communicator waiting for data...")
        while True:
            # Read input from stdin
            if conn is None:
                input_data = input()
            else:
                input_data = conn.recv()

            # Exit the loop if no more input is received
            if input_data is None or input_data == "":
                print("NO INPUT RECEIVED")
                continue
           
            # print("Communicator Received input:", input_data)
            if input_data == "exit":
                break

            # Process the input 
            try:
                input_pred = int(input_data["prediction"])
                timestamp = input_data["timestamp"]
                if int(input_pred) not in range(NUM_CLASSES): 
                    input_pred = 0
                gesture = gjutils.get_label_from_index(input_pred, MEDIA_PATH)

                print(f"Input: pred({input_pred})  gest[{gesture}]: {input_data}")
            except Exception as e:
                print("Invalid input. Error: ", e)
                continue

            # Send the gesture to the hand
            hand_comm.send_gesture(gesture)
    except Exception as e:
        print(f"Error communicator: {e}")
    finally:
        hand_comm.disconnect()
        print("Communicator Exiting...")


# CONNECTION HANDLER

def run_process(target, conn: Connection):
    try:
        target(conn)
    except Exception as e:
        print(f"An error occurred in a subprocess: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    try:
            parent_conn, child_conn = Pipe()
            
            p1 = Process(target=run_process, args=(run_predicator, parent_conn))
            p2 = Process(target=run_process, args=(run_communicator, child_conn))
            
            p1.start()
            p2.start()
            
            p1.join()
            p2.join()
    except Exception as e:
        print(f"An error occurred in the main process: {e}")
    finally:
        parent_conn.close()
        child_conn.close()

        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()
