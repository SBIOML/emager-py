
import os
from emager_py.visualization import realtime_GUI
from emager_py.utils.find_usb import find_psoc
from emager_py.streamers import SerialStreamer
from emager_py.realtime_prediction import EmagerRealtimePredictor
from emager_py.visualization.screen_guided_training import ImageListbox
from emager_py.data.data_generator import EmagerDataGenerator

# parameters

# Find path for the model
model_path = "C:\GIT\EMaGer---GetStarted/realtime_documents\model_felix_full.h5"

VIRTUAL = True
BAUDRATE = 1500000

if VIRTUAL:
    PORT = "COM13" #virtual port
    PORT2 = "COM12" #virtual port
    datasetpath = "C:\GIT\Datasets\EMAGER"
    generator_streamer = SerialStreamer(PORT2, BAUDRATE, VIRTUAL)
    data_generator = EmagerDataGenerator(
        generator_streamer, datasetpath, 1000, 50, True 
    )
    emg, lab = data_generator.prepare_data("000", "001")
    thread = data_generator.start()
    print("Data generator thread started")
else:
    PORT = find_psoc()

# Create Streamer
inputStreamer = SerialStreamer(PORT, BAUDRATE, VIRTUAL)
inputStreamer.open()

images = ImageListbox().start()

# Create Class
dataEMG = EmagerRealtimePredictor(inputStreamer, model=model_path, nb_class=len(images))

# Create GUI
gui = realtime_GUI.RealTimeGestureUi(images)

# register the callback
def call_back_gui(data):
    gui.update_label(data['final_pred'])
dataEMG.register_output_callback(call_back_gui)

def call_back_data(data):
    print(data)
    print(f"Final Pred: {data['final_pred']} ({data['time'].strftime('%H:%M:%S')})")
dataEMG.register_output_callback(call_back_data)


try:
    # starts the threads
    dataEMG.start()
    # starts the GUI loop
    gui.run()
    print("Exiting GUI")

finally:
    dataEMG.stop()
    inputStreamer.close()
    print("Goodbye!")




