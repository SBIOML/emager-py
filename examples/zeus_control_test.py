from emager_py.control.zeus_control import ZeusControl
from emager_py.control.constants import Gesture
import time

        
if __name__ == "__main__":

    try:
        comm = ZeusControl()
        comm.connect()
        comm.start_tellemetry()
        time.sleep(1)
        comm.stop_tellemetry()
        time.sleep(1)
        comm.send_gesture("No_Motion")
        time.sleep(2)
        comm.send_gesture(Gesture.OK)
        time.sleep(5)
        comm.send_gesture("No_Motion")
        time.sleep(2)
        comm.send_gesture(Gesture.THUMBS_UP)
        time.sleep(5)
        comm.send_gesture("No_Motion")
        time.sleep(2)
        comm.send_gesture(Gesture.PEACE)
        time.sleep(5)
        comm.send_gesture("No_Motion")
        time.sleep(2)
        comm.send_gesture(Gesture.INDEX_EXTENSION)
        time.sleep(5)
        comm.send_gesture("No_Motion")
        time.sleep(2)
        comm.send_gesture(Gesture.HAND_CLOSE)
        time.sleep(5)
        comm.send_gesture("No_Motion")
        time.sleep(2)
        comm.send_gesture(Gesture.HAND_OPEN)
        time.sleep(3)
        step = 100
        for i in range(0, 1000, step):
            comm.send_finger_position(0, max(0, i-200))
            comm.send_finger_position(1, i)
            comm.send_finger_position(2, i)
            comm.send_finger_position(3, i)
            comm.send_finger_position(4, i)
            time.sleep(0.1)

    finally:
        comm.disconnect()
        print("Done")
