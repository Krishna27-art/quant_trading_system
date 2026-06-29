"""
Lock-Free SPSC Ring Buffer for Ultra-Low Latency Inter-Process Communication.
Replaces fcntl kernel locks with atomic/lock-free pointer management.
"""

import contextlib
import multiprocessing.shared_memory
import struct


class SharedRingBuffer:
    """
    A zero-copy ring buffer using multiprocessing.shared_memory.
    Implemented as a lock-free Single-Producer Single-Consumer (SPSC) queue.
    """

    def __init__(
        self, name: str, create: bool = False, size: int = 1024 * 1024, max_msg_size: int = 1024
    ):
        self.name = name
        self.create = create
        self.max_msg_size = max_msg_size
        self.capacity = size // max_msg_size

        # Header size: 8 bytes for head, 8 bytes for tail
        self.header_size = 16
        self.total_size = self.header_size + (self.capacity * max_msg_size)

        try:
            if create:
                self.shm = multiprocessing.shared_memory.SharedMemory(
                    name=name, create=True, size=self.total_size
                )
                # Initialize head and tail to 0
                self.shm.buf[:16] = struct.pack("QQ", 0, 0)
            else:
                self.shm = multiprocessing.shared_memory.SharedMemory(name=name)
        except FileExistsError:
            self.shm = multiprocessing.shared_memory.SharedMemory(name=name)

        # Using cast to map pointers as uint64 arrays for direct native memory assignment
        # This mitigates pack/unpack overhead and supports atomic-like assignment on 64-bit systems
        self._head_view = self.shm.buf[0:8].cast("Q")
        self._tail_view = self.shm.buf[8:16].cast("Q")

    def write(self, data: bytes) -> bool:
        if len(data) > self.max_msg_size - 4:
            raise ValueError(
                f"Data size {len(data)} exceeds max message size {self.max_msg_size - 4}"
            )

        head = self._head_view[0]
        tail = self._tail_view[0]

        if head - tail >= self.capacity:
            return False  # Buffer full

        idx = head % self.capacity
        offset = self.header_size + (idx * self.max_msg_size)

        # Write size then data
        self.shm.buf[offset : offset + 4] = struct.pack("I", len(data))
        self.shm.buf[offset + 4 : offset + 4 + len(data)] = data

        # Increment head using the casted memoryview (lock-free)
        self._head_view[0] = head + 1
        return True

    def read(self) -> bytes | None:
        head = self._head_view[0]
        tail = self._tail_view[0]

        if head == tail:
            return None  # Buffer empty

        idx = tail % self.capacity
        offset = self.header_size + (idx * self.max_msg_size)

        # Read size then data
        size = struct.unpack("I", self.shm.buf[offset : offset + 4])[0]
        data = bytes(self.shm.buf[offset + 4 : offset + 4 + size])

        # Increment tail using the casted memoryview (lock-free)
        self._tail_view[0] = tail + 1
        return data

    def close(self):
        self._head_view.release()
        self._tail_view.release()
        self.shm.close()

    def unlink(self):
        try:
            self._head_view.release()
            self._tail_view.release()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            self.shm.unlink()


TICK_FMT = "16s d d d"


class SPSCTickRingBuffer(SharedRingBuffer):
    def write_tick(self, symbol: str, price: float, volume: float, timestamp: float) -> bool:
        sym_bytes = symbol.encode("utf-8").ljust(16, b"\x00")
        payload = struct.pack(TICK_FMT, sym_bytes, price, volume, timestamp)
        return self.write(payload)

    def read_tick(self) -> tuple[str, float, float, float] | None:
        data = self.read()
        if not data:
            return None
        sym_bytes, price, vol, ts = struct.unpack(TICK_FMT, data)
        return sym_bytes.decode("utf-8").rstrip("\x00"), price, vol, ts


ORDER_FMT = "36s 16s q"


class SPSCOrderRingBuffer(SharedRingBuffer):
    def write_order(self, order_id: str, symbol: str, quantity: int) -> bool:
        oid_bytes = order_id.encode("utf-8").ljust(36, b"\x00")
        sym_bytes = symbol.encode("utf-8").ljust(16, b"\x00")
        payload = struct.pack(ORDER_FMT, oid_bytes, sym_bytes, quantity)
        return self.write(payload)

    def read_order(self) -> tuple[str, str, int] | None:
        data = self.read()
        if not data:
            return None
        oid_bytes, sym_bytes, qty = struct.unpack(ORDER_FMT, data)
        return (
            oid_bytes.decode("utf-8").rstrip("\x00"),
            sym_bytes.decode("utf-8").rstrip("\x00"),
            qty,
        )
