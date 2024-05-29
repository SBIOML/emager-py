from emager_py.streamers import SerialStreamer
from emager_py.utils.find_usb import find_psoc
from emager_py.visualisation.color_matrix import RealTimeMatrixPlot

PORT = find_psoc()
stream_client = SerialStreamer(PORT, 1500000)
rt_plot = RealTimeMatrixPlot(stream_client, colormap='plasma', interval=300, min=300, max=30000)
rt_plot.show()
