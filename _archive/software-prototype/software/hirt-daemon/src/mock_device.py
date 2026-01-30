import time
import struct
import random
import threading
import math
from dataclasses import dataclass

# Mock Packet Structure (matches Arch Spec)
# struct DataPacket {
#     uint8_t sync_byte;      // 0xAA
#     uint8_t msg_type;       // 0x01 = Measurement
#     uint16_t tx_id;
#     uint16_t rx_id;
#     uint32_t freq_hz;
#     float    amplitude;
#     float    phase;
#     float    noise_floor;
#     uint16_t crc16;
# };

@dataclass
class Packet:
    tx_id: int
    rx_id: int
    freq_hz: int
    amplitude: float
    phase: float
    noise: float

    def pack(self) -> bytes:
        payload = struct.pack(
            '<BBHHIfff',
            0xAA,           # Sync
            0x01,           # Msg Type
            self.tx_id,
            self.rx_id,
            self.freq_hz,
            self.amplitude,
            self.phase,
            self.noise
        )
        crc = 0xFFFF # Mock CRC
        return payload + struct.pack('<H', crc)

class MockESP32:
    def __init__(self, port='/tmp/ttyHIRT', baud=921600):
        self.port = port
        self.running = False
        print(f"Virtual ESP32 initialized on {port}")

    def generate_scan_data(self):
        """Generates realistic-ish MIT data"""
        t = time.time()
        # Simulate a target at TX=5, RX=6
        base_amp = 100.0
        
        packet = Packet(
            tx_id=random.randint(1, 20),
            rx_id=random.randint(1, 20),
            freq_hz=10000,
            amplitude=base_amp + random.gauss(0, 1.0),
            phase=math.pi / 4 + random.gauss(0, 0.1),
            noise=0.5
        )
        return packet.pack()

    def run(self):
        self.running = True
        print("ESP32 Started. Streaming data...")
        try:
            while self.running:
                data = self.generate_scan_data()
                # In a real mock, we'd write to a PTY
                # For now, just print hex to verify struct
                print(f"TX: {data.hex()}")
                time.sleep(0.1) # 10Hz update
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    device = MockESP32()
    device.run()
