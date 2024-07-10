
import threading
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

import os
import torch
import time
import numpy as np
from multiprocessing import Lock, Process

def update_labels_process(gui:realtime_gui.RealTimeGestureUi, stop_event:threading.Event):
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
        prdeicted_class = int(latest_output[1])
        gui.update_label(prdeicted_class)
        time.sleep(0.1)

if __name__ == "__main__":

    eutils.set_logging()


    MODEL_PATH = "C:\GIT\Datasets\Libemg\Demo\libemg_torch_cnn_Demo_919.pth"
    MEDIA_PATH = "./media-test/"

    NUM_CLASSES = 5
    WINDOW_SIZE=200
    WINDOW_INCREMENT=10
    MAJORITY_VOTE=7

    VIRTUAL = False

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
    oclassi = OnlineEMGClassifier(classi, WINDOW_SIZE, WINDOW_INCREMENT, odh, fg, std_out=True, smm=True, smm_items=smm_items)


    # Create GUI
    files = [MEDIA_PATH + f for f in os.listdir(MEDIA_PATH) if os.path.isfile(os.path.join(MEDIA_PATH, f)) and (f.endswith('.png') or f.endswith('.jpg'))]
    print("Files: ", files)
    print("Creating GUI...")
    gui = realtime_gui.RealTimeGestureUi(files)
    
    stop_event = threading.Event()
    updateLabelProcess = threading.Thread(target=update_labels_process, args=(gui, stop_event))

    try:
        oclassi.run(block=False)
        time.sleep(2)
        updateLabelProcess.start()
        gui.run()
        
    except Exception as e:
        print(f"Error during classification: {e}")

    finally :
        stop_event.set()
        oclassi.stop_running()

        print("Exiting")