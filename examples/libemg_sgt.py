if __name__ == "__main__":
    from libemg.data_handler import OnlineDataHandler
    from libemg.datasets import OneSubjectMyoDataset
    from libemg.gui import GUI
    from libemg.streamers import emager_streamer
    import time
    from emager_py.utils.find_usb import *

    VIRTUAL = True
    SESSION = "Test5"
    DATAFOLDER = f"C:\GIT\Datasets\Libemg\{SESSION}/"

    if VIRTUAL:
        DATASET_PATH = "C:\GIT\Datasets\Libemg\Demo/"
        NUM_CLASSES = 5
        NUM_REPS = 5
        SAMPLING = 1007
        PORT = virtual_port_libemg_v2(SAMPLING, DATASET_PATH, NUM_CLASSES, NUM_REPS)
        print("Data generator thread started")
        time.sleep(3)
    else:
        PORT = None

    # Create data handler and streamer
    p, smi = emager_streamer(specified_port=PORT)
    print(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)
    print("Data handler created")

    args = {
        "online_data_handler": odh,
        "streamer":p,
        "media_folder": "media-test/",
        "data_folder": DATAFOLDER,
        "num_reps" : 5,
        "rep_time": 5,
    }
    gui = GUI(args=args, debug=False, width=900, height=800)
    gui.download_gestures([2,3,10,14,18], "media-test/", download_gifs=False)
    gui.start_gui()