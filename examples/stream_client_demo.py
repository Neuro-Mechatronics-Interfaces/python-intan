import time
from intan.control import IntanRHXClient

client = IntanRHXClient('127.0.0.1', n_channels=1)
client.set_channel('a', 0, enable_wide=True)
client.set_blocks_per_write(1)

client.start_streaming(rate=4000)

def print_channels():
    while True:
        ch_data = client.get_latest_sample()
        print(f"Channel 0: {ch_data[0]}")
        time.sleep(0.05)

print_channels()
