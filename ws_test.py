# ws_test.py
import asyncio
import websockets
import json
import sys

async def run(project_id, user_id):
    uri = f"ws://localhost:8000/ws/{project_id}/{user_id}"
    print("Connecting to", uri)
    async with websockets.connect(uri) as ws:
        async def receiver():
            try:
                while True:
                    msg = await ws.recv()
                    print(f"[{user_id}] RECV:", msg)
            except Exception as e:
                print("Receiver exception:", e)

        async def pinger():
            try:
                while True:
                    await asyncio.sleep(20)
                    await ws.send(json.dumps({"type":"ping"}))
            except Exception as e:
                pass

        await asyncio.gather(receiver(), pinger())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ws_test.py <project_id> <user_id>")
    else:
        asyncio.run(run(sys.argv[1], sys.argv[2]))