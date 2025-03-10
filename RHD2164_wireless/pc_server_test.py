import asyncio
import time


async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"New client connected: {addr}")
    try:
        while True:
            data = await reader.read(1024)
            if not data:
                break
            lines = data.decode().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    print(f"Received from {addr}: {line}")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error with {addr}: {e}")
    print(f"Client disconnected: {addr}")
    writer.close()
    await writer.wait_closed()

async def main():
    host = "192.168.137.1" # hotspot IP
    port = 5001
    server = await asyncio.start_server(handle_client, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Serving on {addrs}")
    async with server:
        await server.serve_forever()


        # time elapsed using the counter
        time.perf_counter()

if __name__ == "__main__":
    asyncio.run(main())
