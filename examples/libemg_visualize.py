if __name__ == "__main__":
    from libemg.data_handler import OnlineDataHandler
    from libemg.streamers import emager_streamer
    from libemg.filtering import Filter
    import time
    from emager_py.utils.find_usb import virtual_port

    VIRTUAL = False
    FILTER = True

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

    if FILTER:
        filter = Filter(1000)
        notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
        filter.install_filters(notch_filter_dictionary)
        bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
        filter.install_filters(bandpass_filter_dictionary)
        odh.install_filter(filter)

    try :
        odh.visualize(num_samples=5000, block=True)
    except Exception as e:
        print(e)
    finally:
        print("Exiting...")