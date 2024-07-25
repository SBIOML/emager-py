from emager_py.control.ble_client import BLEDevice, scan_and_connect
from emager_py.control.serial_com import SerialCommunication
import time

SERVICE_UART = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHAR_UART_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
CHAR_UART_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

class SmartHandControl:

    def __init__(self, deviceName="NewHandCommDevice", mode="BOTH", baud_rate=115200, port=None):
        self.deviceName = deviceName
        self.use_ble = False
        self.device:BLEDevice = None
        self.use_serial = False
        self.serial:SerialCommunication = None
        self.baud_rate = baud_rate
        self.port = port

        mode = mode.upper()
        if "BLE" in mode or "BLUETOOTH" in mode or "BT" in mode or "WIRELESS" in mode or "BOTH" in mode:
            self.use_ble = True
        elif "SERIAL" in mode or "UART" in mode or "USB" in mode or "CABLE" in mode or "BOTH" in mode:
            self.use_serial = True
        else:
            raise ValueError("Invalid mode. Use 'BLE' or 'SERIAL' or 'BOTH' as mode.")
        self.mode = mode

    def connect(self):
        print("Connecting HandComm")
        if self.use_ble:
            self.device = scan_and_connect(self.deviceName, retry=2)
            if self.device:
                self.device.add_characteristic(SERVICE_UART, CHAR_UART_TX)
                self.device.add_characteristic(SERVICE_UART, CHAR_UART_RX)
                self.device.start_notify(CHAR_UART_TX)
        if self.use_serial:
            self.serial = SerialCommunication(port=self.port, baud_rate=self.baud_rate)
            self.serial.open()

    def read_data(self):
        value = b''
        if self.device and self.use_ble:
            value = self.device.read(SERVICE_UART, CHAR_UART_TX)
        if self.serial and self.use_serial:
            value = self.serial.read()
        
        formatted_hex = ' '.join(f'{byte:02x}' for byte in memoryview(value))
        print(f"UART TX: {formatted_hex} \n --> {value.decode()}")

    def send_data(self, data):
        if self.device and self.use_ble:
            self.device.write(SERVICE_UART, CHAR_UART_RX, data)
        if self.serial and self.use_serial:
            self.serial.write(data)

    def send_gesture(self, gesture):
        data =  b'\x02'
        bytes_val = int(gesture).to_bytes(1, 'big')
        data += bytes_val
        self.send_data(data)

    def send_finger_position(self, finger, position):
        data =  b'\x01'
        bytes_val_finger = int(finger).to_bytes(1, 'big')
        data += bytes_val_finger
        bytes_val_pos = int(position).to_bytes(2, 'big')
        data += bytes_val_pos
        self.send_data(data)

    def toggle_led_rpi(self):
        self.send_data("toggle")  

    def blink_led_rpi(self):
        self.send_data("blink")

    def disconnect(self):
        if self.device and self.use_ble:
            self.device.stop_notify(CHAR_UART_TX)
            self.device.disconnect()
        if self.serial and self.use_serial:
            self.serial.close()