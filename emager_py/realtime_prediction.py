import numpy as np
import time, threading, datetime
import tensorflow as tf
import collections
from scipy import signal
from emager_py.streamers import EmagerStreamerInterface

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
    zi = np.tile(zi, (64, 1))
    return b_combined, a_combined, zi


class HDEMG(object):
    def __init__(self, inputStreamer:EmagerStreamerInterface, model, nb_class=5, window_size=30, 
                 batch_size=80, majority_size=250, sample_buffer_size=100, filter_buffer_size=100, 
                 input_buffer_size=200, prediction_buffer_size=200):
        
        self.inputStreamer = inputStreamer
        self._outputs_calbacks = []
        
        # parameters
        self.nb_class = nb_class
        self.window_size = window_size
        self.batch_size = batch_size
        self.majority_size = majority_size

        self.sample_buffer_size = sample_buffer_size
        self.filter_buffer_size = filter_buffer_size
        self.input_buffer_size = input_buffer_size
        self.prediction_buffer_size = prediction_buffer_size

        self.old_avg = 1 - (1/self.window_size)
        self.new_avg = 1/self.window_size
        self.b, self.a, self.zi = preprocess_init()
        self.t1 = 0
        self.final_pred = 0

        # Load NN model
        self.model = tf.keras.models.load_model(model + ".h5",compile=False)
        self.model.compile()

        # create all the deque buffers
        self.sample_buffer = collections.deque(maxlen=self.sample_buffer_size)
        self.filter_buffer = collections.deque(maxlen=self.filter_buffer_size)
        self.window_buffer = np.zeros((1, 64))
        self.input_buffer = collections.deque(maxlen=self.input_buffer_size)
        self.batch_buffer = np.zeros((self.batch_size,4,16,1))
        self.prediction_buffer = collections.deque(maxlen=self.prediction_buffer_size)
        self.majority_buffer = collections.deque(maxlen=self.majority_size)

    def sample_data(self):
        countSamples = 0
        self.inputStreamer.clear() 

        while True:
            read_data = self.inputStreamer.read()
            if read_data is None or len(read_data) == 0:
                continue
            data = np.array(read_data)

            if countSamples >= 10000:
                self.t1 = time.time()
                countSamples = countSamples - 10000

            countSamples += data.shape[0]
            self.sample_buffer.extend(data)

    def preprocess(self):
        y = np.zeros((64,))
        while True:
            if self.sample_buffer:
                if len(self.sample_buffer) <= 64:
                    time.sleep(0.002)
                    continue
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
                if len(self.filter_buffer) <= 0:
                    time.sleep(0.002)
                    continue
                data = self.filter_buffer.popleft()
                if len(self.window_buffer) == self.window_size:
                    self.window_buffer = np.delete(self.window_buffer, 0, axis=0)
                self.window_buffer = np.vstack((self.window_buffer, data))
                data = np.mean(self.window_buffer, axis=0)
                self.input_buffer.append(data)
            else:
                time.sleep(0.002)


    def cnn_predict(self):
        i = 0
        while True:
            if self.input_buffer:
                if len(self.input_buffer) <= 0:
                    time.sleep(0.002)
                    continue
                sample = self.input_buffer.popleft()
                sample = sample.reshape((4, 16, 1))
                self.batch_buffer[i] = sample
                i += 1
                if i == self.batch_size:
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
        while True:
            if self.prediction_buffer:
                if len(self.prediction_buffer) <= 0:
                    time.sleep(0.002)
                    continue
                pred = self.prediction_buffer.popleft()
                self.majority_buffer.append(pred)
                pred_nb +=1
                for n in range(self.nb_class):
                    count[n] = self.majority_buffer.count(n)
                    if count[n] > hysterisis[n]:
                        self.final_pred = n
                        # now = datetime.datetime.now()
                        # print(F"Final Pred: {self.final_pred} ({now.strftime('%H:%M:%S')})")
                
            else:
                time.sleep(0.002)

    def register_output_callback(self, callback):
        if callback not in self._outputs_calbacks:
            self._outputs_calbacks.append(callback)

    def unregister_output_callback(self, callback):
        if callback in self._outputs_calbacks:
            self._outputs_calbacks.remove(callback)

    def _notify_output_callbacks(self, data):
        for callback in self._outputs_calbacks:
            callback(data)

    def output_data(self):
        while True:
            output = {
                'final_pred': self.final_pred,
                'time': datetime.datetime.now()
            }
            self._notify_output_callbacks(output)
            time.sleep(0.1)


    def start(self):
        process_1 = threading.Thread(target=self.sample_data)
        process_2 = threading.Thread(target=self.preprocess)
        process_3 = threading.Thread(target=self.moving_average)
        process_4 = threading.Thread(target=self.cnn_predict)
        process_5 = threading.Thread(target=self.majority_vote)
        process_6 = threading.Thread(target=self.output_data)

        process_6.start()
        process_5.start()
        process_4.start()
        process_3.start()
        process_2.start()
        process_1.start()

    def stop(self):
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

if __name__=='__main__' :
    import os
    from emager_py.visualisation import realtime_GUI
    from emager_py.utils.find_usb import find_psoc
    from emager_py.streamers import SerialStreamer

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

    # Create GUI
    gui = realtime_GUI.RealTimeGestureUi(5)

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


