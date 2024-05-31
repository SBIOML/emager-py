from emager_py.data_processings.data_generator import EmagerDataGenerator
from emager_py.streamers import TcpStreamer
import emager_py.data_processings.emager_redis as er
import numpy as np

from emager_py.utils import utils
utils.set_logging()

batch = 10
host = er.get_docker_redis_ip()

datasetpath = "C:\GIT\Datasets\EMAGER"
server_stream = TcpStreamer(4444, listen=False)
dg = EmagerDataGenerator(
    server_stream, datasetpath , 1000, batch, True
)
emg, lab = dg.prepare_data("004", "001")
dg.serve_data(True)

print("Len of generated data: ", len(dg))
for i in range(len(lab)):
    data, labels = server_stream.read()
    batch = len(labels)
    print(f"Received shape {data.shape}")
    assert np.array_equal(data, emg[batch * i : batch * (i + 1)]), print(
        "Data does not match."
    )
    assert np.array_equal(labels, lab[batch * i : batch * (i + 1)]), print(
        labels, lab[batch * i : batch * (i + 1)]
    )