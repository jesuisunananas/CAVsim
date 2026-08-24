"""Bridges co-perception's local annotated-frame broadcast (see
output/broadcast_sink.py) to browsers over a WebSocket.

Deliberately a separate process from process_video.py rather than a
thread inside it -- same reasoning the camera pipeline already applies to
splitting demux/decode/upload_aws: a bug or restart in the browser-facing
server shouldn't be able to take down the detection pipeline, and vice
versa. process_video.py never needs to know a browser exists; it just
broadcasts locally, same as decode does for upload_aws.

Each browser message is binary: 1 byte channel index + JPEG bytes for
that channel's latest annotated frame (see common/protocol.py).

Usage: python3 ws_broadcast_server.py [config_path]
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import websockets

from co_perception import config as pipeline_config
from co_perception.common.broadcast import BroadcastClient
from co_perception.common.protocol import MSG_ANNOTATED_FRAME

WS_HOST = "127.0.0.1"  # Loopback only -- nginx (location /perception/ws) is
# what actually faces the internet, same as v2x-drive's own backend on 8765.
WS_PORT = 8766


def local_broadcast_reader(socket_path, loop, clients, clients_lock):
    """Runs on its own thread: blocks on BroadcastClient reads (the local
    Unix socket transport is synchronous) and hands each frame to every
    connected browser's own queue. Each client's queue holds only its
    latest undelivered frame (see _queue_latest) -- a slow or half-dead
    browser connection drops stale frames instead of an unbounded backlog
    of scheduled sends piling up in memory, which is what let this
    process grow to 52GB anon-rss and get OOM-killed after ~5 days
    (2026-08-23 incident)."""
    while True:
        try:
            client = BroadcastClient(socket_path)
        except (ConnectionError, OSError):
            print(f"ws_broadcast_server: waiting for {socket_path} (is process_video.py running with output.broadcast.enabled?)")
            time.sleep(1.0)
            continue

        print(f"ws_broadcast_server: connected to {socket_path}")
        try:
            for msg_type, payload in client:
                if msg_type != MSG_ANNOTATED_FRAME:
                    continue
                with clients_lock:
                    targets = list(clients.values())
                for queue in targets:
                    loop.call_soon_threadsafe(_queue_latest, queue, payload)
        except ConnectionError:
            pass
        client.close()
        print(f"ws_broadcast_server: {socket_path} disconnected, reconnecting")


def _queue_latest(queue, payload):
    """Keep only the newest frame per client: if the previous one hasn't
    been sent yet, drop it rather than let anything accumulate. Must run
    on the event loop thread (scheduled via call_soon_threadsafe) --
    asyncio.Queue isn't thread-safe to touch directly from the reader
    thread."""
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(payload)


async def _sender_loop(ws, queue):
    """One per connected browser: sends whatever's newest in its queue,
    waiting for the next frame once caught up. A client stuck behind is
    always at most one frame's backlog, never unbounded."""
    while True:
        payload = await queue.get()
        await ws.send(payload)


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "config" / "pipeline.yaml")
    cfg = pipeline_config.load_config(config_path, REPO_ROOT)
    if not cfg.broadcast.enabled:
        print("ws_broadcast_server: output.broadcast.enabled is false in config -- nothing to bridge, exiting")
        return

    clients = {}  # ws -> asyncio.Queue(maxsize=1), latest-frame-only per client
    clients_lock = threading.Lock()
    loop = asyncio.get_running_loop()

    reader_thread = threading.Thread(
        target=local_broadcast_reader,
        args=(cfg.broadcast.socket_path, loop, clients, clients_lock),
        daemon=True,
    )
    reader_thread.start()

    async def handler(ws):
        queue = asyncio.Queue(maxsize=1)
        with clients_lock:
            clients[ws] = queue
        print(f"ws_broadcast_server: browser connected ({len(clients)} total)")
        try:
            await _sender_loop(ws, queue)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with clients_lock:
                clients.pop(ws, None)
            print(f"ws_broadcast_server: browser disconnected ({len(clients)} total)")

    async with websockets.serve(handler, WS_HOST, WS_PORT):
        print(f"ws_broadcast_server: listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
