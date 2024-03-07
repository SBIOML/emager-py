import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt
import threading

import fabric
from invoke import Responder

import emager_py.utils as eutils
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
        num_columns = 4

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
        self.timer.setTimerType(Qt.PreciseTimer)
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
        print(new_data.shape)
        nb_pts = new_data.shape[1]

        if nb_pts == 0:
            return

        self.tot_samples += nb_pts
        t = time.time()
        print(
            f"({t-self.t0:.3f} s) nsamples= {self.tot_samples}, dt={t - self.timestamp:.3f} s"
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
        self.app.exec_()


def run_remote_finn(conn: fabric.Connection, script: str):
    """
    Run a remote Python script on PYNQ which uses FINN. On PYNQ, the ran script is `run.sh`,
    which takes in a single argument, the python file to execute.

    If there are some PYNQ errors, maybe some more stuff needs to be sourced from, PYNQ's `/etc/profile.d/`.

    Example: `self.run_remote_finn(c, "validate_finn.py")` runs `bash run.sh validate_finn.py` on the remote PYNQ.
    Assumes sudo password is `xilinx`.

    Returns whatever `conn.run(...)` returns.
    """
    sudopass = Responder(
        pattern=r"\[sudo\] password for .*:",
        response="xilinx\n",
    )
    result = conn.run(
        f"bash /home/xilinx/workspace/pynq-emg/run.sh {script}",
        pty=True,
        watchers=[sudopass],
    )
    return result


if __name__ == "__main__":
    import emager_py.data_generator as edg

    eutils.set_logging()

    GENERATE = True
    HOST = "172.17.0.2" if GENERATE else "pynq"
    # HOST = "pynq"

    dg = edg.EmagerDataGenerator(
        HOST, "/home/gabrielgagne/Documents/git/emager-pytorch/data/EMAGER/", False
    )

    dg.r.flushall()
    dg.r.set(eutils.GENERATED_SAMPLES_KEY, 1000 * 3600)
    dg.r.set(eutils.FS_KEY, 1000)
    dg.r.set(eutils.BATCH_KEY, 10)
    dg.r.set(
        eutils.BITSTREAM_KEY,
        b"/home/xilinx/workspace/pynq-emg/bitfile/finn-accel.bit",
    )
    dg.r.set("rhd_enable_dsp", 0)

    if GENERATE:
        dg.prepare_data("004", "001")
        dg.update_params()
        dg.get_serve_thread().start()
    else:
        c = fabric.Connection("xilinx@pynq", connect_kwargs={"password": "xilinx"})
        t = threading.Thread(
            target=run_remote_finn, args=(c, "rhd-sampler/build/rhd_sampler")
        ).start()

    streamer = streamers.RedisStreamer(HOST)
    oscilloscope = RealTimeOscilloscope(streamer, 64, 1000, 3, 30)

    oscilloscope.run()
