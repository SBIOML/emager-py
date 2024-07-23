import asyncio
from emager_py.communication.ble_client import BLEDevice, scan_and_connect
import time

SERVICE_UART = "6E400001-C352-11E5-953D-0002A5D5C51B"
CHAR_UART_TX = "6E400003-C352-11E5-953D-0002A5D5C51B"
CHAR_UART_RX = "6E400002-C352-11E5-953D-0002A5D5C51B"
NAME = "A-235328"

class ZeusControl:

    def __init__(self, deviceName=NAME):
        self.deviceName = deviceName
        self.device:BLEDevice = None
        self.crc32 = CRC32()

    def connect(self):
        print("Connecting ZeusHand")
        self.device = scan_and_connect(self.deviceName, retry=2)
        if self.device:
            self.device.add_characteristic(SERVICE_UART, CHAR_UART_TX)
            self.device.add_characteristic(SERVICE_UART, CHAR_UART_RX)
            self.device.add_notification_callback(self._notify_callback)
            self.device.start_notify(CHAR_UART_TX)

    def disconnect(self):
        if self.device:
            self.device.stop_notify(CHAR_UART_TX)
            self.device.disconnect()
        self.device = None
        print("Disconnected ZeusHand")


    def _read_data_packet(self, packet):
        if len(packet) < 8:
            return None, None, "Invalid packet length"
        
        amber_spp_header = packet[0]
        frame_header = packet[1:3]
        checksum = int.from_bytes(packet[3:7], byteorder='big')
        frame_type = packet[7]
        frame_data = packet[8:]
        
        if amber_spp_header != 0x01 or frame_header != bytes([0xA5, 0x5A]):
            return None, None, "Invalid header"
        
        # Calculate the expected CRC32 for the frameType and frameData
        data_for_crc = packet[7:]
        calculated_checksum = self.crc32.soft_crc32_from_buffer(data_for_crc)
        if calculated_checksum != checksum:
            return None, None, "Checksum mismatch"
        
        return frame_type, frame_data, "Success"
    
    def _write_data_packet(self, frame_type, frame_data):
        amber_spp_header = bytes([0x01])
        frame_header = bytes([0xA5, 0x5A])
        frame_type_byte = bytes([frame_type])
        frame_data_bytes = bytes(frame_data)
        
        # Combine frame type and frame data for checksum calculation
        data_for_crc = frame_type_byte + frame_data_bytes
        
        # Calculate checksum using the CRC32 instance
        checksum = self.crc32.soft_crc32_from_buffer(data_for_crc)
        checksum_bytes = checksum.to_bytes(4, byteorder='big')
        
        # Construct the packet
        packet = amber_spp_header + frame_header + checksum_bytes + frame_type_byte + frame_data_bytes
        
        return packet
    
    def _notify_callback(self, sender, data, args):
        frame_type, frame_data, status = self._read_data_packet(data)
        if status == "Success":
            print(f"Notification Received frame type: {frame_type}, frame data: {frame_data}")
        else:
            print(f"Notification Error: {status}")

    def read_data(self):
        value = b''
        if self.device:
            value = self.device.read(SERVICE_UART, CHAR_UART_TX)
            print(f"Reading Received: {value}")
            # formatted_hex = ' '.join(f'{byte:02x}' for byte in memoryview(value))
            # print(f"UART TX: {formatted_hex} \n --> {value.decode()}")
            frame_type, frame_data, status = self._read_data_packet(value)
            if status == "Success":
                print(f"Reading Received frame type: {frame_type}, frame data: {frame_data}")
            else:
                print(f"Reading Error: {status}")

    def send_data(self, data, data_id=None):
        if self.device:
            packet = self._write_data_packet(data_id, data)
            self.device.write(SERVICE_UART, CHAR_UART_RX, packet)

    def send_gesture(self, gesture):
        self.send_data(gesture, data_id=0x09)

    def send_finger_position(self, finger, position):
        finger_bytes = int(finger).to_bytes(1, 'big')
        position_bytes = int(position).to_bytes(4, 'big')
        data = finger_bytes + position_bytes
        self.send_data(data, data_id=0x05)


# For Packet Validation
class CRC32:
    def __init__(self):
        self.crc32_table = self.init_crc32_table()

    @staticmethod
    def init_crc32_table():
        polynomial = 0xEDB88320
        crc32_table = []
        for index in range(256):
            crc = index
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ polynomial
                else:
                    crc >>= 1
            crc32_table.append(crc)
        return crc32_table

    def soft_crc32_from_buffer(self, buffer):
        current_value = 0xFFFFFFFF
        for byte in buffer:
            table_index = (current_value ^ byte) & 0xFF
            current_value = (current_value >> 8) ^ self.crc32_table[table_index]
        return ~current_value & 0xFFFFFFFF
        
if __name__ == "__main__":

    comm = ZeusControl()
    comm.connect()
    time.sleep(5)
    comm.read_data()
    time.sleep(5)
    comm.send_gesture(6)
    time.sleep(5)
    while True:
        comm.read_data()
        time.sleep(5)
    comm.disconnect()

    print("Done")

