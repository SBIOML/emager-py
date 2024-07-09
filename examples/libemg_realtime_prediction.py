if __name__ == "__main__":
    from libemg.data_handler import OnlineDataHandler
    from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier
    from libemg.feature_extractor import FeatureExtractor
    from libemg.streamers import emager_streamer
    from libemg.filtering import Filter

    from emager_py.utils.find_usb import virtual_port
    import emager_py.torch.models as etm

    import torch
    import time
    import numpy as np

    import emager_py.utils.utils as eutils

    eutils.set_logging()


    MODEL_PATH = "C:\GIT\Datasets\Libemg\Test3\libemg_torch_cnn_Test3_708.pth"

    NUM_CLASSES = 5
    WINDOW_SIZE=200
    WINDOW_INCREMENT=10
    MAJORITY_VOTE=5

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
    oclassi = OnlineEMGClassifier(classi, WINDOW_SIZE, WINDOW_INCREMENT, odh, fg, std_out=True, smm=False)
    try:
        oclassi.run(block=True)
    except Exception as e:
        print(f"Error during classification: {e}")
        # Handle specific classification errors here

    try :
        while True:
            time.sleep(1)
    except Exception as e:
        print("Exception: ", e)
    finally :
        print("Exiting")