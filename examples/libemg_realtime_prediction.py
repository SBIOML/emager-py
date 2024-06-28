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


    model_path = "emager_torch_cnn_0_1.pth"
    window_size=30
    window_increment=1
    majority_size=250
    NUM_CLASSES = 6
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
    print("Data handler created")

    # Choose feature group and classifier
    fe = FeatureExtractor()
    fg = fe.get_feature_groups()["HTD"]
    print("Feature group: ", fg)

    # Load model
    model = etm.EmagerCNN((4, 16), NUM_CLASSES, -1)
    model.load_state_dict(torch.load(model_path))
    classi = EMGClassifier()
    classi.add_majority_vote(majority_size)
    classi.classifier = model.eval()
    print("Model loaded")

    oclassi = OnlineEMGClassifier(classi, window_size, window_increment, odh, fg, std_out=True, smm=False)
    oclassi.run(block=True)
    print("Online classifier started")

  