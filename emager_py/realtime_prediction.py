import numpy as np
import time, threading, datetime
import tensorflow as tf
import SensorLib
import collections
from scipy import signal
import realtime_GUI
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from multiprocessing import Process
from matplotlib import pyplot as plt
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def preprocess_init():
    # high pass (DC offset removal)
    fs = 1010  # Sampling frequency
    f_cutoff = 10  # Cutoff frequency
    order = 2
    b_DC, a_DC = signal.butter(order, f_cutoff, fs=fs, btype='high', analog=False, output='ba')

    # 60 Hz filter
    fs = 1010  # Sampling frequency
    f0 = 60  # Filter frequency
    Q = 30  # Quality factor
    b_60, a_60 = signal.iirnotch(f0, Q, fs)

    # combine as and bs
    b_combined = np.polymul(b_DC, b_60)
    a_combined = np.polymul(a_DC, a_60)
    zi = signal.lfilter_zi(b_combined, a_combined)
    #zi = signal.lfilter_zi(b_DC, a_DC)
    zi = np.tile(zi, (64, 1))
    return b_combined, a_combined, zi


class HDEMG(object):
    def __init__(self, model, serialpath, baudrate):
        # Creates Sensor Object
        self.sensor = SensorLib.HDSensor(serialpath, baudrate)
        self.sensor.open()

        # parameters
        self.nb_class = 5
        self.window = 30
        self.batchsize = 80
        self.majority = 250
        self.old_avg = 1 - (1/self.window)
        self.new_avg = 1/self.window
        self.b, self.a, self.zi = preprocess_init()
        self.t1 = 0
        self.final_pred = 0

        # Load NN model
        # is_gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0
        # print(is_gpu)
        # print(tf.debugging.set_log_device_placement(True))
        # gpu = tf.config.experimental.list_physical_devices('GPU')
        # print("GPU:", gpu)
        # tf.config.experimental.set_memory_growth(gpu, True)
        self.model = tf.keras.models.load_model(model + ".h5",compile=False)
        self.model.compile()
        # Load the min and max for MinMaxScaling
        # min_and_max = np.loadtxt(model + ".csv", delimiter=',')
        # self.min = min_and_max[:, 0]
        # self.max = min_and_max[:, 1]

        # create all the deque buffers
        self.sample_buffer = collections.deque(maxlen=100)
        self.filter_buffer = collections.deque(maxlen=100)
        self.window_buffer = np.zeros((1, 64))
        self.input_buffer = collections.deque(maxlen=200)
        self.batch_buffer = np.zeros((self.batchsize,4,16,1))
        self.prediction_buffer = collections.deque(maxlen=200)
        self.majority_buffer = collections.deque(maxlen=self.majority)

    def sample_intan(self):
        first = True
        sensor_count = 0
        self.sensor.clear_buffer()
        while True:
            data = np.array(self.sensor.read_full_buffer())
            if sensor_count >= 10000:
                self.t1 = time.time()
                sensor_count = sensor_count-10000

            sensor_count += data.shape[0]
            # print(nb_instance)
            self.sample_buffer.extend(data)
            # print(f"sample_intan: {self.sample_buffer} " )

    def preprocess(self):
        y = np.zeros((64,))
        while True:
            #print("preprocess:",len(self.sample_buffer))
            if self.sample_buffer:
                raw_data = self.sample_buffer.popleft()
                for k in range(64):
                    y_new, zi_new = signal.lfilter(self.b, self.a, [raw_data[k]], zi=self.zi[k, :])
                    self.zi[k, :] = zi_new
                    y[k] = np.abs(y_new)
                self.filter_buffer.append(y)
            else:
                time.sleep(0.002)


    def moving_average(self): #MinMaxScale as well
        while True:
            if self.filter_buffer:
                #print("moving average:", len(self.filter_buffer))
                data = self.filter_buffer.popleft()
                if len(self.window_buffer) == self.window:
                    self.window_buffer = np.delete(self.window_buffer, 0, axis=0)
                self.window_buffer = np.vstack((self.window_buffer, data))
                data = np.mean(self.window_buffer, axis=0)
                #data = (data - self.min) / (self.max - self.min) # Optional
                self.input_buffer.append(data)
            else:
                time.sleep(0.002)

    def cnn_predict(self):
        i = 0
        while True:
            #print(len(self.input_buffer))
            if self.input_buffer:
                sample = self.input_buffer.popleft()
                sample = sample.reshape((4, 16, 1))
                self.batch_buffer[i] = sample
                i += 1
                if i == self.batchsize:
                    prediction = np.argmax(self.model.predict_on_batch(self.batch_buffer),axis=1)
                    self.prediction_buffer.extend(prediction)
                    i = 0
            else:
                time.sleep(0.002)


    def majority_vote(self):
        hysterisis = [210,130,130,220,130]
        hysterisis = [210, 130, 130, 130, 200]  # 0:fist, 1:pouce, 2:neutre, 3:pince, 4: doigt
        count = np.zeros((6,1))
        pred_nb = 0
        pred_memory = []
        while True:
            #print(len(self.prediction_buffer))
            if self.prediction_buffer:
                pred = self.prediction_buffer.popleft()
                #pred_memory.append(pred)
                #print(pred)
                self.majority_buffer.append(pred)
                pred_nb +=1
                for n in range(self.nb_class):
                    count[n] = self.majority_buffer.count(n)
                    if count[n] > hysterisis[n]:
                        self.final_pred = n

                now = datetime.datetime.now()
                # self.final_pred = np.argmax(count)
                print(F"Final Pred: {self.final_pred} ({now.strftime('%H:%M:%S')})")
            else:
                time.sleep(0.002)
            # if pred_nb == 10000:
            #     delay = time.time()-self.t1
            #     #print(pred_memory)
            #     pred_memory = []
            #     #if delay > 0.1:
            #         #print("Excess delay:", delay)
            #     pred_nb = 0
            #     t1 = time.time()



    def run(self):
        process_1 = threading.Thread(target=self.sample_intan)
        process_2 = threading.Thread(target=self.preprocess)
        process_3 = threading.Thread(target=self.moving_average)
        process_4 = threading.Thread(target=self.cnn_predict)
        process_5 = threading.Thread(target=self.majority_vote)
        process_5.start()
        process_4.start()
        process_3.start()
        process_2.start()
        process_1.start()

        # Real-time GUI with PyQt to display detected gestures
        app = QApplication([])
        gui = realtime_GUI.ui(self.nb_class)
        gui.show()
        timer = QTimer()
        print("Initialisation!!")
        timer.timeout.connect(lambda: gui.setImg(self.final_pred))
        timer.start(100)  # display refresh interval (ms)
        app.exec()

    def join_threads(self):
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

if __name__=='__main__' :
    # parameters
    # Find path for the model
    # Find path for the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "AI", "models", "model_felix_with_transfer")
    # Create Class
    dataEMG = HDEMG(model=model_path, serialpath='COM4', baudrate=1500000)

    try:
        # starts the main loop
        dataEMG.run()
        

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Keyboard interrupt received. Terminating subprocesses REAL_TIME_MAIN ...")
        if 'sender_stream' in locals():
            dataEMG.sensor.close()
        dataEMG.join_threads()
        # sys.exit(0)

    finally:
        # Close the serial port
        print("Closing REAL_TIME_MAIN serial port...")
        sys.stdout.flush()
        dataEMG.join_threads()
        # dataEMG.sensor.close()
        # sys.exit(0)






# def sensor_init(model, serialpath, baudrate):
#     # Creates Sensor Object
#     sensor = SensorLib.HDSensor(serialpath, baudrate)
#     sensor.open()
#     return sensor

# def sample_intan(sensor):
#     sensor.clear_buffer()
#     while True:
#         data = np.array(sensor.read_full_buffer())
#         nb_instance = data.shape[0]
#         print(nb_instance)
#
#
# def run(sensor):
#     process_1 = threading.Thread(target=sample_intan, args=(sensor,))
#     process_1.start()
#     print("We runnin! Let's go baby!!")
#
# if __name__=='__main__' :
#     # parameters
#     model_name = 'model_009'
#     # Create Class
#     sensor = sensor_init(model=model_name, serialpath='COM7', baudrate=1500000)
#     # starts the main loop
#     run(sensor)




    # def moving_average(self):
    #     mean_val = np.zeros((64))
    #     while True:
    #         print("moving average:", len(self.filter_buffer))
    #         if self.filter_buffer:
    #             #t1 = time.time()
    #             data = self.filter_buffer.popleft()
    #             mean_val = self.old_avg*mean_val + self.new_avg*data
    #             self.input_buffer.append(data)
    #             #print("time:", time.time() - t1)
    #         else:
    #             time.sleep(0.002)



# import serial, time
# arduino = serial.Serial('COM3', 115200, timeout=.1)
# time.sleep(1) #give the connection a second to settle
# #serialcmd = input("Serial Command:")
# #arduino.write(serialcmd.encode())
# while True:
#   serialcmd = input("Serial Command:")
#   arduino.write(serialcmd.encode())
#   data = arduino.readline() #Reads what the Arduino received and shot back
#   if data:
#     print (data) #strip out the new lines for now
# 		# (better to do .read() in the long run for this reason









