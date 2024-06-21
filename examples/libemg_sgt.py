if __name__ == "__main__":
    from libemg.screen_guided_training import ScreenGuidedTraining
    from libemg.data_handler import OnlineDataHandler
    from libemg.gui import GUI
    from libemg.streamers import emager_streamer
    import time
    from emager_py.utils.find_usb import virtual_port

    VIRTUAL = False

    if VIRTUAL:
        DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
        PORT = virtual_port(DATASET_PATH)
        print("Data generator thread started")
    else:
        PORT = None

    

    time.sleep(3)
    # Create data handler and streamer
    p, smi = emager_streamer(specified_port=PORT)
    print(f"Streamer created: process: {p}, smi : {smi}")
    time.sleep(3)
    odh = OnlineDataHandler(shared_memory_items=smi)
    time.sleep(3)

    args = {
        "online_data_handler": odh,
        "streamer":p
    }
    gui = GUI(args=args, debug=False, width=600, height=500)
    gui.download_gestures([1,2,3,4,5], "gestures/", download_gifs=True)
    gui.start_gui()

    while True:
        time.sleep(1)