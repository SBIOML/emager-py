import serial.tools.list_ports
import emager_py.data.data_generator as edg
import sys
import time
from emager_py.streamers import SerialStreamer, socat_serial_serial

def find_port(vid, pid):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == vid and port.pid == pid:
            print(f"Found device: {port.device}")
            return port.device
    raise ValueError("Device not found")
    return None

def find_psoc():
    return find_port(0x04b4, 0xf155)

def find_pico():
    return find_port(0x2e8a, 0x0005)


def virtual_port(datasetpath, baudrate=1500000, subjectId="000", sessionId="001") -> str:
    PORT1 = '/dev/ttyV1' if sys.platform.startswith('linux') else 'COM1'
    PORT2 = '/dev/ttyV2' if sys.platform.startswith('linux') else 'COM2'

    if sys.platform.startswith('linux'):
        proc = socat_serial_serial(PORT1, PORT2)
        time.sleep(0.5)
    generator_streamer = SerialStreamer(PORT2, baudrate, True)
    data_generator = edg.EmagerDataGenerator(
        generator_streamer, datasetpath, 1000, 50, True 
    )
    emg, lab = data_generator.prepare_data(subjectId, sessionId)
    print(f"Data prepared: {emg.shape}, {lab.shape}")
    thread = data_generator.start()
    return PORT1