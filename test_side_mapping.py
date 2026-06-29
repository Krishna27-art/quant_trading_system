import unittest

from portfolio_execution.execution.brokers.base import OrderSide
from portfolio_execution.orchestrator import TradingOrchestrator


class TestSideMapping(unittest.TestCase):
    def test_side_mapping_never_inverts(self):
        # We just need to initialize the Orchestrator without all dependencies to test mapping.
        # But wait, orchestrator might need dependencies.
        # Since _map_side doesn't depend on self state, we can just call it on a mocked object or pass None.
        class MockOrchestrator(TradingOrchestrator):
            def __init__(self):
                pass

        orch = MockOrchestrator()

        # Test BUY mappings
        self.assertEqual(orch._map_side("BUY"), OrderSide.BUY)
        self.assertEqual(orch._map_side("LONG"), OrderSide.BUY)
        self.assertEqual(orch._map_side("COVER"), OrderSide.BUY)

        # Test SELL mappings
        self.assertEqual(orch._map_side("SELL"), OrderSide.SELL)
        self.assertEqual(orch._map_side("SHORT"), OrderSide.SELL)
        self.assertEqual(orch._map_side("EXIT_LONG"), OrderSide.SELL)

        # Test invalid mapping
        with self.assertRaises(ValueError):
            orch._map_side("HOLD")


if __name__ == "__main__":
    unittest.main()
