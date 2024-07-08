if __name__ == "__main__":
    from libemg.data_handler import OnlineDataHandler
    from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier
    from libemg.feature_extractor import FeatureExtractor
    from libemg.streamers import emager_streamer

    from emager_py.utils.find_usb import virtual_port
    import emager_py.torch.models as etm

    import torch
    import time
    import numpy as np

    import emager_py.utils.utils as eutils

    eutils.set_logging()


    model_path = "C:\GIT\Datasets\EMAGER\emager_torch_cnn_13_2"
    window_size=30
    window_increment=1
    majority_size=250
    NUM_CLASSES = 5
    VIRTUAL = True

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
    print("Data handler created")

    # Choose feature group and classifier
    fe = FeatureExtractor()
    fg = fe.get_feature_groups()["HTD"]
    print("Feature group: ", fg)

    # Verify model loading and state dict compatibility
    model = etm.EmagerCNN((4, 16), NUM_CLASSES, -1)
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        # Handle error (e.g., exit or attempt a recovery)

    classi = EMGClassifier()
    classi.add_majority_vote(majority_size)
    classi.classifier = model.eval()

    # Ensure OnlineEMGClassifier is correctly set up for data handling and inference
    oclassi = OnlineEMGClassifier(classi, window_size, window_increment, odh, fg, std_out=True, smm=True)
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