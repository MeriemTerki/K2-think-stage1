"""Automated simulation script to test the conflict flow.

This script will:
- open two websocket clients (alice and bob) and print messages
- call REST endpoints to start sessions and send file activity
- wait for messages then exit

Usage:
    python simulate_conflict.py

Requires: websockets, httpx
Install with: pip install websockets httpx
"""
import asyncio
import json
import httpx
import websockets

API_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
PROJECT_ID = "project1"


async def ws_listener(user_id):
    uri = f"{WS_BASE}/ws/{PROJECT_ID}/{user_id}"
    async with websockets.connect(uri) as ws:
        print(f"[{user_id}] connected to {uri}")
        async for message in ws:
            print(f"[{user_id}] RECV: {message}")


async def do_rest_calls():
    async with httpx.AsyncClient() as client:
        # start alice
        r = await client.post(f"{API_BASE}/api/session/start", json={
            "user_id": "alice",
            "username": "Alice",
            "project_id": PROJECT_ID,
            "branch": "feature/x"
        })
        print("start alice ->", r.status_code, r.text)

        await asyncio.sleep(1)

        # start bob
        r = await client.post(f"{API_BASE}/api/session/start", json={
            "user_id": "bob",
            "username": "Bob",
            "project_id": PROJECT_ID,
            "branch": "feature/x"
        })
        print("start bob ->", r.status_code, r.text)

        await asyncio.sleep(1)

        # alice edits file
        r = await client.post(f"{API_BASE}/api/file/activity", json={
            "user_id": "alice",
            "username": "Alice",
            "project_id": PROJECT_ID,
            "file_path": "src/a.py",
            "action": "editing",
            "branch": "feature/x"
        })
        print("alice edit ->", r.status_code, r.text)

        await asyncio.sleep(1)

        # bob edits same file -> should trigger conflict_warning
        r = await client.post(f"{API_BASE}/api/file/activity", json={
            "user_id": "bob",
            "username": "Bob",
            "project_id": PROJECT_ID,
            "file_path": "src/a.py",
            "action": "editing",
            "branch": "feature/x"
        })
        print("bob edit ->", r.status_code, r.text)

        await asyncio.sleep(3)

        # optional: simulate push
        r = await client.post(f"{API_BASE}/api/git/operation", json={
            "user_id": "alice",
            "username": "Alice",
            "project_id": PROJECT_ID,
            "operation": "push",
            "branch": "feature/x",
            "files": ["src/a.py"]
        })
        print("alice push ->", r.status_code, r.text)


async def main():
    # start two websocket listeners
    alice_task = asyncio.create_task(ws_listener("alice"))
    bob_task = asyncio.create_task(ws_listener("bob"))

    # give sockets a moment to connect
    await asyncio.sleep(1)

    # perform REST calls
    await do_rest_calls()

    # wait a bit to collect messages
    await asyncio.sleep(5)

    # cancel listeners
    alice_task.cancel()
    bob_task.cancel()

    try:
        await alice_task
    except asyncio.CancelledError:
        pass
    try:
        await bob_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())