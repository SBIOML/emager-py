import serial
import serial.tools.list_ports
from time import sleep, time

class SerialCommunication:
    def __init__(self, port=None, baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None

    def _find_port(self):
        vid, pid = 0x2e8a, 0x0005
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.vid == vid and port.pid == pid:
                print(f"Found device: {port.device}")
                return port.device
        raise ValueError("Device not found")
    
    def open(self):
        if self.port is None:
            self.port = self._find_port()
        if self.serial is None:
            self.serial = serial.Serial(port=self.port, baudrate=self.baud_rate, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE, timeout=1.5)
        if not self.serial.is_open:
            self.serial.open()
        sleep(0.2)

    def close(self):
        if self.serial is not None:
            if self.serial.is_open:
                self.serial.close()

    def write(self, message):
        if isinstance(message, str):
            message = (message + "\n").encode()
        else:
            message = bytes(message)
            message = (message.decode() + "\n").encode()
        if self.serial is not None:
            self.serial.write(message)
            self.read()

    def read(self):
        if self.serial is not None:
            return self.serial.read_until().strip()
