import time
from emager_py.streamers import SerialStreamer
from emager_py.utils.find_usb import find_psoc
from emager_py.visualisation.screen_guided_training import ImageListbox, EmagerGuidedTraining

imgbox = ImageListbox(num_columns=5)
selected_gestures = imgbox.start()
print(f"Selected gestures: {selected_gestures}")

# PORT = find_psoc()
PORT = "COM13" #virtual port
streamer = SerialStreamer(PORT, baud=1500000, virtual=True)

def my_cb(gesture):
    print("Simulating long running process...")
    time.sleep(5)
    print(f"Gesture {gesture+1} done!")

egt = EmagerGuidedTraining(
    streamer, selected_gestures,
    resume_training_callback=my_cb,  callback_arg="gesture", reps=3, training_time=4,
)
egt.start()
print("Exiting...")