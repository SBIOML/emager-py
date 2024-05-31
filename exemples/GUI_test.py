import random
import time
from emager_py.visualisation.screen_guided_training import ImageListbox
from emager_py.visualisation.realtime_GUI import RealTimeGestureUi
import threading

images = ImageListbox().start()

gui = RealTimeGestureUi(images)

class label_class:
    def __init__(self, n=5):
        self.label = 0
        self.n = n

    def change_label(self):
        while gui.isRunning:
            time.sleep(1)
            self.label = random.randint(0, self.n-1)

lc= label_class(n=len(images))
label_class_thread_instance = threading.Thread(target=lc.change_label, args=())
label_class_thread_instance.start()

def label_thread():
    while gui.isRunning:
        gui.update_label(lc.label)
        time.sleep(0.1)
label_thread_instance = threading.Thread(target=label_thread, args=())
label_thread_instance.start()
gui.run()