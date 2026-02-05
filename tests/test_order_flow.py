import unittest
from ml.order_flow_analyzer import OrderFlowAnalyzer

class TestOrderFlowAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = OrderFlowAnalyzer()

    def test_obi_calculation_balanced(self):
        # Balanced book
        orderbook = {
            'bids': [[100, 1], [99, 1], [98, 1], [97, 1], [96, 1]],
            'asks': [[101, 1], [102, 1], [103, 1], [104, 1], [105, 1]]
        }
        obi = self.analyzer.calculate_obi(orderbook)
        self.assertAlmostEqual(obi, 0.0)

    def test_obi_calculation_buy_pressure(self):
        # Heavy bids
        orderbook = {
            'bids': [[100, 10], [99, 10]],
            'asks': [[101, 1], [102, 1]]
        }
        obi = self.analyzer.calculate_obi(orderbook)
        self.assertGreater(obi, 0.5)

    def test_obi_calculation_sell_pressure(self):
        # Heavy asks
        orderbook = {
            'bids': [[100, 1], [99, 1]],
            'asks': [[101, 10], [102, 10]]
        }
        obi = self.analyzer.calculate_obi(orderbook)
        self.assertLess(obi, -0.5)

    def test_toxicity_calculation(self):
        # Add some trades
        # Trade 1: Buy @ 100. Trade 11 (Future): 105 (>100) -> Adverse for Seller?
        # Analyzer logic:
        # If I am MM, and Aggressor BUYS, and price goes UP, I sold low and it went up. ADVERSE.
        # Logic in code:
        # if side == 'buy' and future_price > entry_price: is_adverse = True
        
        # 0..19 trades
        for i in range(20):
            price = 100 + i
            # Trade i
            self.analyzer.calculate_toxicity({'price': price, 'side': 'buy'})
            
        # Check toxicity
        # Trade 0 (100) -> Future Trade 10 (110). 110 > 100. Adverse.
        # Trade 5 (105) -> Future Trade 15 (115). 115 > 105. Adverse.
        # Should be 100% toxicity for this linear uptrend
        
        tox = self.analyzer.calculate_toxicity(None)
        print(f"Calculated Toxicity: {tox}")
        self.assertGreater(tox, 0.8)

if __name__ == '__main__':
    unittest.main()
