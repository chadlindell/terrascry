import time
import struct
import logging
import argparse
import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hirt-daemon')

@dataclass
class ScanConfig:
    survey_name: str = "test_survey"
    operator: str = "hirt_operator"
    site_id: str = "site_001"

class HirtDaemon:
    def __init__(self, data_dir="data", mock_mode=True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mock_mode = mock_mode
        self.current_h5 = None
        self.running = False
        
        # Packet Structure (matches Firmware Spec)
        # sync(1) + type(1) + tx(2) + rx(2) + freq(4) + amp(4) + ph(4) + noise(4) + crc(2) = 24 bytes
        self.PACKET_SIZE = 24
        self.SYNC_BYTE = 0xAA

    def start_new_survey(self, config: ScanConfig):
        """Creates a new HDF5 file for the survey"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = self.data_dir / f"{config.survey_name}_{timestamp}.h5"
        
        logger.info(f"Creating new survey file: {filename}")
        self.current_h5 = h5py.File(filename, 'w')
        
        # Write Metadata
        self.current_h5.attrs['version'] = 1.0
        self.current_h5.attrs['operator'] = config.operator
        self.current_h5.attrs['site_id'] = config.site_id
        self.current_h5.attrs['start_time'] = timestamp
        
        # Create Groups
        self.grp_raw = self.current_h5.create_group("raw_data")
        
        # Create Resizable Datasets
        # We use a chunked layout to allow appending data indefinitely
        self.dset_mit = self.grp_raw.create_dataset(
            "mit_measurements",
            shape=(0,),
            maxshape=(None,),
            dtype=[
                ('timestamp', 'f8'),
                ('tx_id', 'u2'),
                ('rx_id', 'u2'),
                ('freq_hz', 'u4'),
                ('amplitude', 'f4'),
                ('phase', 'f4'),
                ('noise', 'f4')
            ],
            chunks=True
        )
        return filename

    def process_packet(self, data: bytes):
        """Parses binary packet and writes to HDF5"""
        try:
            # Unpack struct (Little Endian <)
            sync, msg_type, tx, rx, freq, amp, phase, noise = struct.unpack(
                '<BBHHIfff', data[:22]  # Exclude CRC for now
            )
            
            if sync != self.SYNC_BYTE:
                logger.warning(f"Invalid Sync Byte: {hex(sync)}")
                return

            if msg_type == 0x01: # Measurement
                # Append to HDF5
                # Note: Resizing dataset every sample is slow in production, 
                # but fine for this "Phase 1" proof of concept.
                # Production would buffer 1000 samples then write.
                
                size = self.dset_mit.shape[0]
                self.dset_mit.resize((size + 1,))
                
                record = (
                    time.time(),
                    tx, rx, freq, amp, phase, noise
                )
                self.dset_mit[size] = record
                
                if size % 10 == 0:
                    logger.debug(f"Recorded: TX={tx} RX={rx} Amp={amp:.2f}")
                    self.current_h5.flush() # Ensure data is written
                    
        except struct.error as e:
            logger.error(f"Packet unpack error: {e}")

    def run(self, mock_device=None):
        """Main Loop"""
        self.running = True
        logger.info("Daemon started. Waiting for data...")
        
        try:
            while self.running:
                if self.mock_mode and mock_device:
                    # Direct function call from mock script (for testing)
                    data = mock_device.generate_scan_data()
                    self.process_packet(data)
                    time.sleep(0.1) # Simulate 10Hz sample rate
                else:
                    # TODO: Implement real Serial.read() here
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("Stopping daemon...")
        finally:
            if self.current_h5:
                self.current_h5.close()
                logger.info("HDF5 file closed safely.")

if __name__ == "__main__":
    # Integration Test Mode: Run Daemon + Mock Device together
    from mock_device import MockESP32
    
    daemon = HirtDaemon(data_dir="data_out")
    mock = MockESP32()
    
    logger.info("--- STARTING INTEGRATION TEST ---")
    
    # 1. Start Survey
    h5_path = daemon.start_new_survey(ScanConfig())
    
    # 2. Run for 5 seconds
    t_end = time.time() + 5
    while time.time() < t_end:
        data = mock.generate_scan_data()
        daemon.process_packet(data)
        time.sleep(0.05) # 20Hz speedup
        
    daemon.current_h5.close()
    logger.info(f"--- TEST COMPLETE. Data saved to {h5_path} ---")
