if __name__ == "__main__":
    from libemg.data_handler import OnlineDataHandler
    from libemg.datasets import OneSubjectMyoDataset
    from libemg.gui import GUI
    from libemg.streamers import emager_streamer
    import time
    from emager_py.utils.find_usb import virtual_port

    VIRTUAL = False
    SESSION = "TestVideo"
    DATAFOLDER = f"C:\GIT\Datasets\Libemg\{SESSION}/"

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