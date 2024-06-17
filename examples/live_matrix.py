from emager_py.streamers import SerialStreamer
from emager_py.utils.find_usb import find_psoc, virtual_port
from emager_py.visualization.color_matrix import RealTimeMatrixPlot
from emager_py.data.data_generator import EmagerDataGenerator

VIRTUAL = False
BAUDRATE = 1500000

if VIRTUAL:
    datasetpath = "C:\GIT\Datasets\EMAGER/"
    PORT = virtual_port(datasetpath, BAUDRATE)
    print("Data generator thread started")
else:
    PORT = find_psoc()

stream_client = SerialStreamer(PORT, BAUDRATE, VIRTUAL)
rt_plot = RealTimeMatrixPlot(stream_client, colormap='plasma', interval=300, min=300, max=65000)
rt_plot.show()
