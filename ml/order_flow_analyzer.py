from collections import deque
import logging

class OrderFlowAnalyzer:
    """
    Analyzes Order Flow using Order Book Imbalance (OBI) and Trade Flow Toxicity.
    Goal: Detect Adverse Selection risk.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger("OrderFlowAnalyzer")
        
        # OBI Config
        self.obi_levels = self.config.get('obi', {}).get('levels', 5)
        self.obi_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Fixed weights for top 5 levels
        
        # Toxicity Config
        self.toxicity_window = self.config.get('toxicity', {}).get('window', 100)
        self.trades = deque(maxlen=self.toxicity_window + 20)  # + buffer for lookahead

    def calculate_obi(self, orderbook):
        """
        Calculates Weighted Order Book Imbalance.
        Formula: (WeightedBidVol - WeightedAskVol) / TotalWeightedVol
        """
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.0

            # Ensure we don't go out of bounds
            depth = min(len(bids), len(asks), len(self.obi_weights), self.obi_levels)
            
            w_bid_vol = 0.0
            w_ask_vol = 0.0
            
            for i in range(depth):
                w = self.obi_weights[i]
                
                # Handle [price, qty] list format
                # If dict, adapt accordingly (PaperExchange uses dict, Mock uses list)
                bid_q = float(bids[i][1]) if isinstance(bids[i], (list, tuple)) else float(bids[i]['amount'])
                ask_q = float(asks[i][1]) if isinstance(asks[i], (list, tuple)) else float(asks[i]['amount'])
                
                w_bid_vol += bid_q * w
                w_ask_vol += ask_q * w
            
            total_vol = w_bid_vol + w_ask_vol
            if total_vol == 0:
                return 0.0
                
            return (w_bid_vol - w_ask_vol) / total_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating OBI: {e}")
            return 0.0

    def calculate_toxicity(self, current_trade=None):
        """
        Calculates Trade Flow Toxicity based on Adverse Price Excursion (APE).
        Simplified version: % of trades followed by adverse price move within 10 ticks.
        """
        if current_trade:
            self.trades.append(current_trade)
            
        if len(self.trades) < 20:
             return 0.0
             
        # Backtest Toxicity: Iterate over recent history
        adverse_count = 0
        total_checks = 0
        
        # Check last N trades where we have 'future' data (index i+10)
        # Using deque, we scan from start to end-10
        
        trades_list = list(self.trades)
        limit = len(trades_list) - 10
        
        for i in range(limit):
            trade = trades_list[i]
            future_trade = trades_list[i+10]
            
            entry_price = float(trade['price'])
            future_price = float(future_trade['price'])
            side = trade['side'] # 'buy' or 'sell'
            
            is_adverse = False
            if side == 'buy':
                # Buyer pushed price up? No, buyer wants price to go up.
                # If price went DOWN after buy, they got rekt (Adverse Selection).
                # Wait, Flow Toxicity usually measures if the AGGRESSOR was informed.
                # If Aggressor BUYs and price goes UP -> Informed (Toxic to MM selling to them).
                if future_price > entry_price:
                    is_adverse = True
            elif side == 'sell':
                # Aggressor SELLs and price goes DOWN -> Informed.
                if future_price < entry_price:
                    is_adverse = True
            
            if is_adverse:
                adverse_count += 1
            total_checks += 1
            
        if total_checks == 0:
            return 0.0
            
        return adverse_count / total_checks

    def get_adjustment_factors(self, orderbook, trade=None):
        """
        Returns multipliers for Spread and Size based on signals.
        """
        obi = self.calculate_obi(orderbook)
        toxicity = self.calculate_toxicity(trade)
        
        # Defaults
        spread_mult = 1.0
        bid_size_mult = 1.0
        ask_size_mult = 1.0
        
        # 1. OBI Logic
        # OBI > 0 (Buy Pressure) -> Price likely to go UP.
        # MM should be reluctant to SELL (Ask) and eager to BUY (Bid)?
        # Or if pressure is up, we want to follow trend?
        # Standard MM: Skew towards inventory.
        
        # PRD Logic:
        # If OBI > 0 (Buy Pressure): Ask Size down (don't sell cheap), Spread Up.
        if abs(obi) > 0.3:
            spread_mult += abs(obi) * 0.5
            
            if obi > 0: # Buy Pressure
                ask_size_mult = 1.0 - (obi * 0.3) # Reduce Ask Size
            else: # Sell Pressure
                bid_size_mult = 1.0 - (abs(obi) * 0.3) # Reduce Bid Size
                
        # 2. Toxicity Logic
        if toxicity > 0.6:
            spread_mult += (toxicity - 0.6) * 2.0 # Aggeressive spread widening
            bid_size_mult *= 0.8
            ask_size_mult *= 0.8
            
        return {
            'spread_mult': min(spread_mult, 2.0),
            'bid_size_mult': max(bid_size_mult, 0.1),
            'ask_size_mult': max(ask_size_mult, 0.1),
            'metrics': {'obi': obi, 'toxicity': toxicity}
        }
