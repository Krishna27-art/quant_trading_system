import time


class FIXEngine:
    """
    Correct FIX 4.2 message builder.
    Never uses string replace() for field calculation.
    BodyLength computed from actual byte count.
    Checksum computed from byte sum mod 256.
    """

    SOH = b"\x01"
    FIX_VERSION = b"FIX.4.2"

    def __init__(self, sender: str, target: str):
        self.sender = sender
        self.target = target
        self._seq_num = 0

    def _next_seq(self) -> int:
        self._seq_num += 1
        return self._seq_num

    def _tag(self, num: int, value: bytes) -> bytes:
        return f"{num}=".encode() + value + self.SOH

    def _build(self, msg_type: bytes, body_fields: bytes) -> bytes:
        seq = self._next_seq()
        ts = time.strftime("%Y%m%d-%H:%M:%S").encode()

        header_body = (
            self._tag(49, self.sender.encode())
            + self._tag(56, self.target.encode())
            + self._tag(34, str(seq).encode())
            + self._tag(52, ts)
            + self._tag(35, msg_type)
            + body_fields
        )

        # BodyLength = byte count of everything after BodyLength field
        begin = self._tag(8, self.FIX_VERSION)
        bodylen = self._tag(9, str(len(header_body)).encode())
        full = begin + bodylen + header_body

        checksum = sum(full) % 256
        trailer = self._tag(10, f"{checksum:03d}".encode())
        return full + trailer

    def create_new_order_single(
        self,
        cl_ord_id: str,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        order_type: str,  # 'MARKET' or 'LIMIT'
        quantity: int,
        price: float | None = None,
    ) -> bytes:
        side_map = {"BUY": b"1", "SELL": b"2"}
        type_map = {"MARKET": b"1", "LIMIT": b"2"}

        fields = (
            self._tag(11, cl_ord_id.encode())
            + self._tag(55, symbol.encode())
            + self._tag(54, side_map.get(side, b"1"))
            + self._tag(60, time.strftime("%Y%m%d-%H:%M:%S").encode())
            + self._tag(38, str(quantity).encode())
            + self._tag(40, type_map.get(order_type, b"1"))
        )

        if order_type == "LIMIT" and price is not None:
            fields += self._tag(44, f"{price:.2f}".encode())
            fields += self._tag(59, b"0")  # Day order

        return self._build(b"D", fields)

    def create_cancel_order(
        self, orig_cl_ord_id: str, cl_ord_id: str, symbol: str, side: str, quantity: int
    ) -> bytes:
        side_map = {"BUY": b"1", "SELL": b"2"}
        fields = (
            self._tag(41, orig_cl_ord_id.encode())
            + self._tag(11, cl_ord_id.encode())
            + self._tag(55, symbol.encode())
            + self._tag(54, side_map.get(side, b"1"))
            + self._tag(38, str(quantity).encode())
            + self._tag(60, time.strftime("%Y%m%d-%H:%M:%S").encode())
        )
        return self._build(b"F", fields)

    def parse_execution_report(self, raw: bytes) -> dict:
        """Parse incoming FIX message into dict."""
        fields = {}
        for field in raw.split(self.SOH):
            if b"=" in field:
                tag, value = field.split(b"=", 1)
                fields[tag.decode()] = value.decode()
        return fields

    def validate_checksum(self, raw: bytes) -> bool:
        if b"10=" not in raw:
            return False
        body, chk = raw.rsplit(b"10=", 1)
        expected = sum(body + b"10=") % 256
        received = int(chk.rstrip(self.SOH))
        return expected == received
