import time
import numpy as np
import threading
import logging as log

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt

import emager_py.streamers as streamers


class RealTimeOscilloscope:
    def __init__(
        self,
        data_server: streamers.EmagerStreamerInterface,
        n_ch: int,
        signal_fs: float,
        accumulate_t: float,
        refresh_rate: float,
    ):
        """
        Create the Oscilloscope.

        - data_server: implements read() method, returning a (n_samples, n_ch) array. Must return (0, n_ch) when no samples are available.
        - n_ch: number of "oscilloscope channels"
        - signal_fs: Signal sample rate [Hz]
        - accumulate_t: x-axis length [s]
        - refresh_rate: oscilloscope refresh rate [Hz]
        """
        self.n_ch = n_ch
        self.data_points = int(accumulate_t * signal_fs)
        self.refresh_rate = refresh_rate
        self.server = data_server

        # Create a time axis
        self.t = np.linspace(0, accumulate_t, self.data_points)
        # Initialize the data buffer for each signal
        self.data = [np.zeros(self.data_points) for _ in range(n_ch)]

        # Create the application
        self.app = QApplication([])

        # Create a window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle("Real-Time Oscilloscope")
        self.win.setBackground(QtGui.QColor(255, 255, 255))  # white: (255, 255, 255)

        # Define the number of rows and columns in the grid
        num_rows = 16

        # Create plots for each signal
        self.plots = []
        for i in range(n_ch):
            row = i % num_rows
            col = i // num_rows
            p = self.win.addPlot(row=row, col=col)
            p.setYRange(-10000, 10000)
            p.getAxis("left").setStyle(
                showValues=False
            )  # Remove axis title, keep axis lines
            p.getAxis("bottom").setStyle(
                showValues=False
            )  # Remove axis title, keep axis lines
            graph = p.plot(self.t, self.data[i], pen=pg.mkPen(color="r", width=2))
            self.plots.append(graph)
        # Set up a QTimer to update the plot at the desired refresh rate
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // refresh_rate)
        self.t2 = time.time()
        self.win.show()

        self.timestamp = time.time()
        self.t0 = time.time()
        self.tot_samples = 0

    def update(self):
        new_data = np.zeros((0, self.n_ch))
        while True:
            # Fetch all available data
            tmp_data = self.server.read()
            if tmp_data.shape[0] == 0:
                break
            new_data = np.vstack((new_data, tmp_data))
        new_data = np.transpose(new_data)
        nb_pts = new_data.shape[1]

        if nb_pts == 0:
            return

        self.tot_samples += nb_pts
        t = time.time()
        log.info(
            f"({t-self.t0:.3f} s), new data shape={new_data.shape}, total samples={self.tot_samples}, dt={t - self.timestamp:.3f} s"
        )
        self.timestamp = t

        for i in range(self.n_ch):
            self.data[i] = np.roll(self.data[i], -nb_pts)  # Shift the data
            # self.data[i][-nb_pts:] = signal.decimate(new_data[i],2)  # Add new data point
            self.data[i][-nb_pts:] = new_data[i]

        for i, plot_item in enumerate(self.plots):
            plot_item.setData(self.t, self.data[i])
        # print("time update:", time.time()-self.t1)

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    import emager_py.data_generator as edg
    import emager_py.emager_redis as er
    import emager_py.utils as eutils
    import emager_py.finn.remote_operations as ro

    eutils.set_logging()

    GENERATE = False
    HOST = er.get_docker_redis_ip() if GENERATE else "pynq"
    # HOST = "pynq"

    r = er.EmagerRedis(HOST)
    dg = edg.EmagerDataGenerator(
        HOST, "/home/gabrielgagne/Documents/git/emager-pytorch/data/EMAGER/", False
    )

    r.clear_data()
    r.set_sampling_params(1000, 10, 1000000)
    r.set_pynq_params("None")
    r.set_rhd_sampler_params(
        15, 350, 0, 0, ro.DEFAULT_EMAGER_PYNQ_PATH + "/bitfile/finn-accel.bit"
    )

    if GENERATE:
        dg.update_params()
        dg.prepare_data("004", "001")
        dg.get_serve_thread().start()
    else:
        c = ro.connect_to_pynq()
        t = threading.Thread(
            target=ro.run_remote_finn,
            args=(c, ro.DEFAULT_EMAGER_PYNQ_PATH, "rhd-sampler/build/rhd_sampler"),
        ).start()

    streamer = streamers.RedisStreamer(HOST)
    oscilloscope = RealTimeOscilloscope(streamer, 64, r.get_int(r.FS_KEY), 3, 30)
    oscilloscope.run()
