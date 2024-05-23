import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt
from emager_py.SensorLib import reorder, HDSensor
from scipy import signal
from emager_py.utils.find_usb import find_psoc

class RealTimeOscilloscope:
    def __init__(self, num_signals, data_points, refresh_rate):
        self.num_signals = num_signals
        self.data_points = data_points
        self.refresh_rate = refresh_rate
        self.firstGo = True

        # Create a time axis
        self.t = np.linspace(0, 3, data_points) # .astype(object)

        # Create the application
        self.app = QApplication([])

        # Create a window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Real-Time Oscilloscope')
        self.win.setBackground(QtGui.QColor(255,255,255)) #white: (255, 255, 255)
        self.win.show()

        # Initialize the data buffer for each signal
        self.data = [np.zeros(data_points) for _ in range(num_signals)]

        # Define the number of rows and columns in the grid
        num_rows = 16
        num_columns = 4

        # Create plots for each signal
        self.plots = []
        for i in range(num_signals):
            row = i % num_rows
            col = i // num_rows
            p = self.win.addPlot(row=row, col=col)
            p.setYRange(-10000, 10000)
            p.getAxis('left').setStyle(showValues=False)  # Remove axis title, keep axis lines
            p.getAxis('bottom').setStyle(showValues=False)  # Remove axis title, keep axis lines
            graph = p.plot(self.t, self.data[i], pen=pg.mkPen(color='r', width=2))
            self.plots.append(graph)
        # Set up a QTimer to update the plot at the desired refresh rate
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // refresh_rate)
        self.t2 = time.time()

    def update(self):
        # print("time between interrupts:", time.time() - self.t2)
        # self.t1= time.time()
        # self.t2= time.time()
        new_data, nb_pts = sensor.live_read(firstTime=self.firstGo, decimate=False)
        # nb_pts = nb_pts//2 + nb_pts%2
        self.firstGo = False
        if nb_pts != 0:
            for i in range(self.num_signals):
                self.data[i] = np.roll(self.data[i], -nb_pts)  # Shift the data
                # self.data[i][-nb_pts:] = signal.decimate(new_data[i],2)  # Add new data point
                self.data[i][-nb_pts:] = new_data[i]
            y_values = [self.data[i] for i in range(self.num_signals)]
            for i, plot_item in enumerate(self.plots):
                plot_item.setData(self.t, y_values[i])
            # print("time update:", time.time()-self.t1)


    def run(self):
        if __name__ == '__main__':
            self.app.exec()


if __name__ == '__main__':
    firstGo = True
    port = find_psoc()
    sensor = HDSensor(port, 1500000)
    num_signals = 64
    data_points = 2000  # 3 seconds at 100 samples per second
    refresh_rate = 30  # 30Hz refresh rate
    oscilloscope = RealTimeOscilloscope(num_signals, data_points, refresh_rate)
    oscilloscope.run()






# import time
# import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import QTimer, Qt
# from SensorLib import reorder, HDSensor
# from scipy import signal
#
# class RealTimeOscilloscope:
#     def __init__(self, num_signals, data_points, refresh_rate):
#         self.num_signals = num_signals
#         self.data_points = data_points
#         self.refresh_rate = refresh_rate
#         self.firstGo = True
#
#         # Create a time axis
#         self.t = np.linspace(0, 3, data_points)#.astype(object)
#
#         # Create the application
#         self.app = QApplication([])
#
#         # Create a window
#         self.win = pg.GraphicsLayoutWidget()
#         self.win.setWindowTitle('Real-Time Oscilloscope')
#         self.win.setBackground(QtGui.QColor(255,255,255)) #white: (255, 255, 255)
#         self.win.show()
#
#         # Initialize the data buffer for each signal
#         self.data = [np.zeros(data_points) for _ in range(num_signals)]
#
#         # Define the number of rows and columns in the grid
#         num_rows = 16
#         num_columns = 4
#
#         # Create plots for each signal
#         self.plots = []
#         for i in range(num_signals):
#             row = i % num_rows
#             col = i // num_rows
#             p = self.win.addPlot(row=row, col=col)
#             p.setYRange(-10000, 10000)
#             p.getAxis('left').setStyle(showValues=False)  # Remove axis title, keep axis lines
#             p.getAxis('bottom').setStyle(showValues=False)  # Remove axis title, keep axis lines
#             graph = p.plot(self.t, self.data[i], pen=pg.mkPen(color='r', width=2))
#             self.plots.append(graph)
#         # Set up a QTimer to update the plot at the desired refresh rate
#         self.timer = QTimer()
#         self.timer.setTimerType(Qt.PreciseTimer)
#         self.timer.timeout.connect(self.update)
#         self.timer.start(1000 // refresh_rate)
#         self.t2 = time.time()
#
#     def update(self):
#         # print("time between interrupts:", time.time() - self.t2)
#         # self.t1= time.time()
#         # self.t2= time.time()
#         new_data, nb_pts = sensor.live_read(firstTime=self.firstGo, decimate=False)
#         # nb_pts = nb_pts//4 + nb_pts%4
#         self.firstGo = False
#         if nb_pts != 0:
#             for i in range(self.num_signals):
#                 self.data[i] = np.roll(self.data[i], -nb_pts)  # Shift the data
#                 # self.data[i][-nb_pts:] = signal.decimate(new_data[i],4)  # Add new data point
#                 self.data[i][-nb_pts:] = new_data[i]
#                 self.plots[i].setData(self.t, self.data[i])
#             # print("time update:", time.time()-self.t1)
#
#
#     def run(self):
#         if __name__ == '__main__':
#             self.app.exec_()
#
#
# if __name__ == '__main__':
#     firstGo = True
#     sensor = HDSensor('COM13', 1500000)
#     num_signals = 64
#     data_points = 2000  # 3 seconds at 100 samples per second
#     refresh_rate = 30  # 30Hz refresh rate
#     oscilloscope = RealTimeOscilloscope(num_signals, data_points, refresh_rate)
#     oscilloscope.run()








# import time
# import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import QTimer, QObject, pyqtSlot, Qt
# from SensorLib import reorder, HDSensor
#
# class PreciseTimer(QObject):
#     def __init__(self, interval, timeout_callback):
#         super().__init__()
#
#         self.timer = QTimer()
#         self.timer.setTimerType(Qt.PreciseTimer)
#         self.interval = interval
#         self.timeout_callback = timeout_callback
#
#         self.start()
#
#     def start(self):
#         self.timer.timeout.connect(self.handle_timeout)
#         self.timer.start(self.interval)
#
#     @pyqtSlot()
#     def handle_timeout(self):
#         elapsed_time = self.timer.remainingTime()
#         self.timeout_callback(elapsed_time)
#
# class RealTimeOscilloscope:
#     def __init__(self, num_signals, data_points, refresh_rate):
#         self.num_signals = num_signals
#         self.data_points = data_points
#         self.refresh_rate = refresh_rate
#
#         # Create a time axis
#         self.t = np.linspace(0, 3, data_points)  # .astype(object)
#
#         # Create the application
#         self.app = QApplication([])
#
#         # Create a window
#         self.win = pg.GraphicsLayoutWidget()
#         self.win.setWindowTitle('Real-Time Oscilloscope')
#         self.win.setBackground(QtGui.QColor(255, 255, 255))  # white: (255, 255, 255)
#         self.win.show()
#
#         # Initialize the data buffer for each signal
#         self.data = [np.zeros(data_points) for _ in range(num_signals)]
#
#         # Define the number of rows and columns in the grid
#         num_rows = 16
#         num_columns = 4
#
#         # Create plots for each signal
#         self.plots = []
#         for i in range(num_signals):
#             row = i % num_rows
#             col = i // num_rows
#             p = self.win.addPlot(row=row, col=col)
#             p.setYRange(-10000, 10000)
#             p.getAxis('left').setStyle(showValues=False)  # Remove axis title, keep axis lines
#             p.getAxis('bottom').setStyle(showValues=False)  # Remove axis title, keep axis lines
#             graph = p.plot(self.t, self.data[i], pen=pg.mkPen(color='r', width=2))
#             self.plots.append(graph)
#
#         # Set up a PreciseTimer to update the plot at the desired refresh rate
#         self.timer = PreciseTimer(1000 // refresh_rate, self.update)
#
#         self.t2 = time.time()
#
#     def update(self, elapsed_time):
#         global firstGo
#         print("time between interrupts:", elapsed_time)
#         self.t2 = time.time()
#         new_data, nb_pts = sensor.live_read(firstTime=firstGo)
#         firstGo = False
#         if nb_pts != 0:
#             for i in range(self.num_signals):
#                 self.data[i] = np.roll(self.data[i], -nb_pts)  # Shift the data
#                 self.data[i][-nb_pts:] = new_data[i]  # Add new data point
#                 self.plots[i].setData(self.t, self.data[i])
#
#     def run(self):
#         if __name__ == '__main__':
#             self.app.exec_()
#
# if __name__ == '__main__':
#     firstGo = True
#     sensor = HDSensor('COM13', 1500000)
#     num_signals = 64
#     data_points = 3000  # 3 seconds at 100 samples per second
#     refresh_rate = 30  # 30Hz refresh rate
#
#     oscilloscope = RealTimeOscilloscope(num_signals, data_points, refresh_rate)
#     oscilloscope.run()