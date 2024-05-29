import time
from emager_py.streamers import SerialStreamer
from emager_py.utils.find_usb import find_psoc
from emager_py.visualisation.screen_guided_training import EmagerGuidedTraining


port = find_psoc()
streamer = SerialStreamer(port)


def my_cb(gesture):
    print("Simulating long running process...")
    time.sleep(5)
    print(f"Gesture {gesture+1} done!")

egt = EmagerGuidedTraining(
    streamer, resume_training_callback=my_cb, reps=1, training_time=1, callback_arg="gesture"
)
egt.start()
print("Exiting...")