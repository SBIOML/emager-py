import time
from emager_py.streamers import SerialStreamer
from emager_py.utils.find_usb import find_psoc, virtual_port
from emager_py.visualization.screen_guided_training import ImageListbox, EmagerGuidedTraining
from emager_py.data import data_generator as dg
from emager_py.utils import utils

utils.set_logging()

imgbox = ImageListbox()
selected_gestures = imgbox.start()
print(f"Selected gestures: {selected_gestures}")


BAUDRATE = 1500000
VIRTUAL = True

if VIRTUAL:
    DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
    PORT = virtual_port(DATASET_PATH, BAUDRATE)
    print("Data generator thread started")
else:
    PORT = find_psoc()

print(f"Streamer on {PORT}")
streamer = SerialStreamer(PORT, BAUDRATE, VIRTUAL)

def my_cb(gesture):
    print("Simulating long running process...")
    time.sleep(5)
    print(f"Gesture {gesture+1} done!")

egt = EmagerGuidedTraining(
    streamer, selected_gestures,
    resume_training_callback=my_cb,  callback_arg="gesture", reps=3, training_time=4,
)
print("Starting guided training...")
egt.start()
print("Exiting...")