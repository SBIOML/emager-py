import random
import time
from emager_py.visualization.screen_guided_training import ImageListbox
from emager_py.visualization.realtime_gui import RealTimeGestureUi
import threading
import emager_py.utils.gestures_json as gjutils

images = ImageListbox().start()

gui = RealTimeGestureUi(images)
stop_gui_event = threading.Event()

class label_class:
    def __init__(self, n=5):
        self.label = 0
        self.n = n

    def change_label(self, stop_event: threading.Event):
        while not stop_event.is_set():
            time.sleep(1)
            self.label = random.randint(0, self.n-1)

lc= label_class(n=len(images))
label_class_thread_instance = threading.Thread(target=lc.change_label, args=(stop_gui_event,))
label_class_thread_instance.start()

def label_thread(stop_event: threading.Event):
    while not stop_event.is_set():
        jlabel = gjutils.get_label_from_index(lc.label, images)
        gui.update_index(lc.label)
        time.sleep(0.1)
label_thread_instance = threading.Thread(target=label_thread, args=(stop_gui_event,))
label_thread_instance.start()

gui.run()
stop_gui_event.set()
print("Exiting GUI")