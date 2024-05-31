import emager_py.finn.remote_operations as ro
import emager_py.data_processings.emager_redis as er
import threading
import time
from emager_py.visualisation.visualize import RealTimeOscilloscope
import emager_py.data_processings.data_generator as edg
import emager_py.utils.utils as eutils
from emager_py.streamers import TcpStreamer

eutils.set_logging()

FS = 1000
BATCH = 1
GENERATE = True
PORT = 12347
HOST = er.get_docker_redis_ip() if GENERATE else "pynq"

def generate_data():
    stream_server = TcpStreamer(PORT, "localhost", True)
    dg = edg.EmagerDataGenerator(
        stream_server,
        "/home/gabrielgagne/Documents/git/emager-pytorch/data/EMAGER/",
        sampling_rate=FS,
        batch_size=BATCH,
        shuffle=False,
    )
    dg.prepare_data("004", "001")
    dg.serve_data()
    print("Started serving data...")

if GENERATE:
    threading.Thread(target=generate_data, daemon=True).start()
else:
    c = ro.connect_to_pynq()
    t = threading.Thread(
        target=ro.run_remote_finn,
        args=(c, ro.DEFAULT_EMAGER_PYNQ_PATH, "rhd-sampler/build/rhd_sampler"),
    ).start()

time.sleep(1)

print("Starting client and oscilloscope...")
stream_client = TcpStreamer(PORT, "localhost", False)


oscilloscope = RealTimeOscilloscope(stream_client, 64, FS, 3, 30)
oscilloscope.run()