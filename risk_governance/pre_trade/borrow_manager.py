import uuid
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class BorrowStatus(str, Enum):
    RESERVED = "RESERVED"
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"


async def reserve_borrow(db: AsyncSession, symbol: str, requested_qty: int) -> tuple[bool, str]:
    """
    Atomic reservation using row-level FOR UPDATE locks.
    Returns (success, message).
    """
    async with db.begin():
        # Lock the row to prevent race conditions
        result = await db.execute(
            text("""
            SELECT available_qty, reserved_qty
            FROM borrow_inventory
            WHERE symbol = :symbol
            FOR UPDATE
        """),
            {"symbol": symbol},
        )

        row = result.fetchone()
        if not row:
            return False, f"No borrow inventory record found for {symbol}"

        available, reserved = row[0], row[1]
        if (available - reserved) >= requested_qty:
            reservation_id = str(uuid.uuid4())
            await db.execute(
                text("""
                UPDATE borrow_inventory
                SET reserved_qty = reserved_qty + :qty, updated_at = now()
                WHERE symbol = :symbol
            """),
                {"qty": requested_qty, "symbol": symbol},
            )

            await db.execute(
                text("""
                INSERT INTO borrow_reservations(reservation_id, symbol, qty, status)
                VALUES (:res_id, :symbol, :qty, :status)
            """),
                {
                    "res_id": reservation_id,
                    "symbol": symbol,
                    "qty": requested_qty,
                    "status": BorrowStatus.RESERVED.value,
                },
            )
            return True, reservation_id
        return (
            False,
            f"Insufficient borrow pool for {symbol}. Needed {requested_qty}, had {available - reserved}.",
        )


async def release_borrow(db: AsyncSession, reservation_id: str) -> bool:
    """Release a reserved borrow back to the pool."""
    async with db.begin():
        # Find the reservation
        result = await db.execute(
            text("""
            SELECT symbol, qty, status
            FROM borrow_reservations
            WHERE reservation_id = :res_id
            FOR UPDATE
        """),
            {"res_id": reservation_id},
        )

        row = result.fetchone()
        if not row or row[2] != BorrowStatus.RESERVED.value:
            return False

        symbol, qty = row[0], row[1]

        # Update inventory
        await db.execute(
            text("""
            UPDATE borrow_inventory
            SET reserved_qty = reserved_qty - :qty, updated_at = now()
            WHERE symbol = :symbol
        """),
            {"qty": qty, "symbol": symbol},
        )

        # Mark reservation as released
        await db.execute(
            text("""
            UPDATE borrow_reservations
            SET status = :status, updated_at = now()
            WHERE reservation_id = :res_id
        """),
            {"res_id": reservation_id, "status": BorrowStatus.RELEASED.value},
        )
        return True


async def release_borrow_by_symbol(db: AsyncSession, symbol: str, qty: int) -> bool:
    """Release the oldest reserved borrow rows for a rejected/cancelled order."""
    remaining = qty
    async with db.begin():
        result = await db.execute(
            text("""
            SELECT reservation_id, qty
            FROM borrow_reservations
            WHERE symbol = :symbol AND status = :status
            ORDER BY reservation_id
            FOR UPDATE
        """),
            {"symbol": symbol, "status": BorrowStatus.RESERVED.value},
        )

        for reservation_id, reserved_qty in result.fetchall():
            if remaining <= 0:
                break
            release_qty = min(int(reserved_qty), remaining)
            await db.execute(
                text("""
                UPDATE borrow_inventory
                SET reserved_qty = reserved_qty - :qty, updated_at = now()
                WHERE symbol = :symbol
            """),
                {"qty": release_qty, "symbol": symbol},
            )
            await db.execute(
                text("""
                UPDATE borrow_reservations
                SET status = :status, updated_at = now()
                WHERE reservation_id = :res_id
            """),
                {"status": BorrowStatus.RELEASED.value, "res_id": reservation_id},
            )
            remaining -= release_qty

    return remaining == 0
