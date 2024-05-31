
import os
from emager_py.visualisation import realtime_GUI
from emager_py.utils.find_usb import find_psoc
from emager_py.streamers import SerialStreamer
from emager_py.realtime_prediction import HDEMG
from emager_py.visualisation.screen_guided_training import ImageListbox

# parameters

# Find path for the model
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = "/home/etienne-michaud/dev/"
model_path = os.path.join(base_dir, "repo_AI_felix", "models", "model_felix_with_transfer")

# Find the port for the PSoC
port = find_psoc()

# Create Streamer
inputStreamer = SerialStreamer(port, 1500000)
inputStreamer.open()

# Create Class
dataEMG = HDEMG(inputStreamer, model=model_path)

images = ImageListbox().start()

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

# starts the threads
dataEMG.start()
# starts the GUI
gui.run()


