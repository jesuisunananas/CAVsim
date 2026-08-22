#!/usr/bin/env python3
"""Local-only pub/sub broadcast over a Unix domain socket: one producer,
N independent consumers on the same host. Verbatim copy of
camera/stream/common/broadcast.py (pure stdlib, no GStreamer dependency,
so it's safe to duplicate rather than import cross-repo -- keeps
co-perception deployable on a machine that doesn't have the camera
project checked out at all). Used here by process_video.py to broadcast
its own annotated output frames to ws_broadcast_server.py.

Design point: a producer with zero connected listeners must not
accumulate anything, and a listener that isn't keeping up must not slow
the producer down either. Every send() is non-blocking and best-effort --
a message that can't be written to a given client's kernel buffer
immediately is never queued or retried; that client is disconnected
instead (see BroadcastServer.send). This keeps RSS flat on the producer
side regardless of whether anyone is listening, matching this project's
whole memory-diagnosis history.

Wire format: each message is a length-prefixed frame:
    1 byte   msg_type (caller-defined, e.g. 1=video, 2=sr_anchor)
    4 bytes  payload length, big-endian unsigned
    N bytes  payload
"""
import os
import selectors
import socket
import struct
import threading
import time

_HEADER = struct.Struct(">BI")  # msg_type, payload_len

# Generous kernel send-buffer per client, and a short bounded window to
# finish a write that didn't complete in one non-blocking call. A partial
# send is normal under real sustained throughput (the kernel buffer being
# momentarily full is not the same thing as a stalled/dead client) -- so
# retrying briefly before giving up avoids disconnecting healthy listeners
# just because they didn't drain instantaneously. A client still behind
# after this window is genuinely not keeping up, and *is* dropped rather
# than queued -- see class docstring.
_CLIENT_SNDBUF_BYTES = 4 * 1024 * 1024
_SEND_RETRY_DEADLINE_SEC = 0.1


class BroadcastServer:
    """One producer. Accepts any number of client connections on a Unix
    domain socket and fans out every send() call to all of them.

    on_client_connected (optional): a zero-arg callable invoked exactly
    once per newly-accepted client, before that client is added to the
    broadcast set. Returning (msg_type, payload) sends that one message
    to just this new client, before anything else it will ever receive
    from a regular send() call; returning None sends nothing extra.
    Exists so demux can hand a late-joining client a cached catch-up
    message (its SPS/PPS) without repeating it for every listener on
    every regular send() -- see demux/main.py."""

    def __init__(self, socket_path, on_client_connected=None):
        self.socket_path = socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(socket_path)
        # connect(2) on a Unix domain socket requires write permission on
        # the socket file itself (see unix(7)) -- default creation mode is
        # umask-dependent and was observed as 0755 (no write for
        # group/other) on this host, which blocks any other local account
        # from subscribing. Set explicitly rather than relying on umask,
        # so every account on this machine can connect regardless of
        # whatever umask the launching process happens to have.
        os.chmod(socket_path, 0o666)
        self._sock.listen(8)
        self._sock.setblocking(False)
        self._clients = []  # list[socket.socket], all non-blocking
        self._lock = threading.Lock()
        self._stopped = False
        self._on_client_connected = on_client_connected
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self):
        sel = selectors.DefaultSelector()
        sel.register(self._sock, selectors.EVENT_READ)
        while not self._stopped:
            for _key, _ in sel.select(timeout=1.0):
                try:
                    conn, _addr = self._sock.accept()
                except OSError:
                    continue
                conn.setblocking(False)
                try:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _CLIENT_SNDBUF_BYTES)
                except OSError:
                    pass  # best-effort; retry-on-partial-write still covers this
                with self._lock:
                    if self._on_client_connected is not None:
                        welcome = self._on_client_connected()
                        if welcome is not None:
                            msg_type, payload = welcome
                            frame = _HEADER.pack(msg_type, len(payload)) + payload
                            if not self._send_all(conn, frame):
                                conn.close()
                                continue
                    self._clients.append(conn)

    def send(self, msg_type, payload):
        """Fan out one message to all currently-connected clients.

        All-or-nothing per client: a client that's still behind after
        _SEND_RETRY_DEADLINE_SEC of retrying is dropped entirely rather
        than sent a partial frame -- a partial frame would desync that
        client's parsing permanently, which is worse than just losing the
        connection and letting it reconnect. But a *single* partial
        write, on its own, is routine under real throughput and not
        treated as fatal -- see _send_all.
        """
        frame = _HEADER.pack(msg_type, len(payload)) + payload
        with self._lock:
            if not self._clients:
                return
            dead = [c for c in self._clients if not self._send_all(c, frame)]
            for c in dead:
                self._clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    @staticmethod
    def _send_all(sock, frame):
        """Send the full frame to one client. Retries a partial write for
        up to _SEND_RETRY_DEADLINE_SEC -- long enough to absorb normal
        kernel-buffer pressure from sustained real throughput, short
        enough that a genuinely stalled client can't hold up the
        producer. Returns False (caller drops the client) if the frame
        still isn't fully sent by the deadline."""
        view = memoryview(frame)
        sent = 0
        deadline = time.monotonic() + _SEND_RETRY_DEADLINE_SEC
        sel = selectors.DefaultSelector()
        sel.register(sock, selectors.EVENT_WRITE)
        try:
            while sent < len(view):
                try:
                    sent += sock.send(view[sent:])
                    continue
                except BlockingIOError:
                    pass
                except OSError:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not sel.select(timeout=remaining):
                    return False
            return True
        finally:
            sel.close()

    def client_count(self):
        with self._lock:
            return len(self._clients)

    def close(self):
        self._stopped = True
        with self._lock:
            for c in self._clients:
                c.close()
            self._clients = []
        self._sock.close()
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass


class BroadcastClient:
    """One consumer. Connects to a BroadcastServer's socket and yields
    (msg_type, payload) tuples as they arrive. A disconnect raises
    ConnectionError from __next__; reconnecting is the caller's own
    responsibility (construct a new BroadcastClient)."""

    def __init__(self, socket_path, connect_timeout=10.0):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(connect_timeout)
        self._sock.connect(socket_path)
        self._sock.settimeout(None)  # blocking reads once connected

    def __iter__(self):
        return self

    def __next__(self):
        header = self._recv_exact(_HEADER.size)
        msg_type, length = _HEADER.unpack(header)
        payload = self._recv_exact(length)
        return msg_type, payload

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("broadcast server closed the connection")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        self._sock.close()


if __name__ == "__main__":
    # Self-test: round-trip a few messages through a real socket, no
    # GStreamer involved, matching this project's script-style validation
    # convention (see stream/decode/rtp_time_calibration_helper.py).
    import tempfile
    import time

    path = os.path.join(tempfile.gettempdir(), "broadcast_selftest.sock")
    server = BroadcastServer(path)
    time.sleep(0.1)
    client = BroadcastClient(path)
    time.sleep(0.1)
    assert server.client_count() == 1

    server.send(1, b"hello")
    server.send(2, struct.pack(">Id", 12345, 1700000000.0))
    msg_type, payload = next(client)
    assert (msg_type, payload) == (1, b"hello")
    msg_type, payload = next(client)
    assert msg_type == 2 and struct.unpack(">Id", payload) == (12345, 1700000000.0)

    client.close()
    time.sleep(0.2)
    server.send(1, b"nobody home")  # must not raise or block with zero listeners
    server.close()

    # on_client_connected: a late joiner gets exactly one welcome message,
    # ahead of anything sent via regular send() calls, and only that client.
    path2 = os.path.join(tempfile.gettempdir(), "broadcast_selftest2.sock")
    welcome_calls = []

    def on_connect():
        welcome_calls.append(1)
        return (9, b"welcome")

    server2 = BroadcastServer(path2, on_client_connected=on_connect)
    time.sleep(0.1)
    client2a = BroadcastClient(path2)
    time.sleep(0.1)
    assert next(client2a) == (9, b"welcome")
    server2.send(1, b"regular")
    assert next(client2a) == (1, b"regular")

    client2b = BroadcastClient(path2)
    time.sleep(0.1)
    assert next(client2b) == (9, b"welcome")
    assert len(welcome_calls) == 2  # once per client, not once total

    client2a.close()
    client2b.close()
    server2.close()

    print("all self-tests passed")
