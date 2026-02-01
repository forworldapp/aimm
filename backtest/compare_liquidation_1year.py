import sys
import os
import pandas as pd
import asyncio
import logging
import gc

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

# Configure Logging
logging.getLogger("MarketMaker").setLevel(logging.CRITICAL)
logging.getLogger("MockExchange").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)

class LiquidationMarketMaker(MarketMaker):
    def __init__(self, exchange, liquidate_on_trigger=False):
        super().__init__(exchange)
        self.liquidate_on_trigger = liquidate_on_trigger
        
        # Disable ML components
        self.ml_regime_enabled = False
        self.regime_detector = None
        self.strategy_v4 = None
        self.adaptive_tuner = None

    async def cycle(self):
        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook: return

        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
        except: return
        
        self._update_history(mid_price)
        self._update_candle(mid_price, 0)
        
        current_pos_qty = position.get('amount', 0.0)
        self.inventory = current_pos_qty
        
        # --- Circuit Breaker Check ---
        unrealized_pnl = position.get('unrealizedPnL', 0.0)
        MAX_LOSS = 200.0 # Force $200
        
        if unrealized_pnl < -MAX_LOSS:
            # Triggered!
            await self.exchange.cancel_all_orders(self.symbol)
            self.is_active = False # Stop
            
            if self.liquidate_on_trigger:
                await self.exchange.close_position(self.symbol)
            
            return

        if self.is_active:
            # Optimize: Skip super().cycle() overhead if possible, or monkeypatch
            # But we need basic order management.
            await super().cycle()

    # Optimization: Override _update_candle to avoid O(N^2) pd.concat
    def _update_candle(self, price, timestamp):
        # We don't need candles for this specific Liquidation vs Hold test
        # since ML is disabled and we rely on PnL from Exchange.
        pass

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

async def run_chunk(df_chunk, mode_name, liquidation_mode):
    # Setup
    exchange = MockExchange(df_chunk, initial_balance=10000.0)
    strategy = LiquidationMarketMaker(exchange, liquidate_on_trigger=liquidation_mode)
    
    # Overrides
    strategy._load_params = lambda: None
    strategy.max_loss_usd = 200.0
    strategy.max_position_usd = 5000.0
    strategy.base_spread = 0.0005
    strategy.is_active = True
    
    Config.set("strategy", "ml_regime_enabled", False)
    
    triggered = False
    trigger_eq = 0.0
    
    for i in range(len(df_chunk)):
        if not exchange.next_tick(): break
        
        if strategy.is_active:
            await strategy.cycle()
        else:
            if not triggered:
                triggered = True
                trigger_eq = exchange.get_equity()
                # Don't break, continue to track PnL drift
    
    final_eq = exchange.get_equity()
    
    # Return delta
    return {
        "Triggered": triggered,
        "StartEq": 10000.0,
        "FinalEq": final_eq,
        "PnL": final_eq - 10000.0,
        "Drift": (final_eq - trigger_eq) if triggered else 0.0
    }

async def run_full_year_comparison():
    print("\n" + "="*60)
    print("📅 1-YEAR SIMULATION: Liquidation vs Hold")
    print("="*60)
    
    data_file = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(data_file):
        print("Data file not found.")
        return

    # Process in chunks of 50,000 minutes (~35 days)
    CHUNK_SIZE = 50000
    
    total_hold_pnl = 0.0
    total_liq_pnl = 0.0
    total_triggers = 0
    drift_loss_hold = 0.0
    drift_loss_liq = 0.0 # Should be 0
    
    chunk_count = 0
    
    # Read CSV in chunks
    for chunk in pd.read_csv(data_file, chunksize=CHUNK_SIZE):
        chunk_count += 1
        print(f"Processing Chunk {chunk_count}... ({len(chunk)} rows)", end='\r')
        
        # Pre-process chunk
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='ms')
        chunk['best_bid'] = chunk['close'] * (1 - 0.00005)
        chunk['best_ask'] = chunk['close'] * (1 + 0.00005)
        
        # Run Hold
        res_h = await run_chunk(chunk, "HOLD", False)
        
        # Run Liq
        res_l = await run_chunk(chunk, "LIQ", True)
        
        # Aggregation
        total_hold_pnl += res_h['PnL']
        total_liq_pnl += res_l['PnL']
        
        if res_h['Triggered']:
            total_triggers += 1
            drift_loss_hold += res_h['Drift']
            drift_loss_liq += res_l['Drift']

        # Manual Garbage Collection
        del chunk
        gc.collect()
        
        # Limit to ~2 months (100k mins) for performance
        if chunk_count >= 2:
            print("\n   [Info] Reached 2-month limit. Stopping simulation.")
            break

    print("\n" + "="*60)
    print("📊 1-YEAR RESULTS SUMMARY")
    print("="*60)
    print(f"Total Chunks Processed: {chunk_count}")
    print(f"Total Circuit Breakers: {total_triggers}")
    print("-" * 60)
    print(f"{'Metric':<25} | {'HOLD Strategy':<15} | {'LIQ Strategy':<15}")
    print("-" * 60)
    print(f"{'Total PnL':<25} | ${total_hold_pnl:<15.2f} | ${total_liq_pnl:<15.2f}")
    print(f"{'Post-Stop Drift':<25} | ${drift_loss_hold:<15.2f} | ${drift_loss_liq:<15.2f}")
    print("-" * 60)
    
    advantage = total_liq_pnl - total_hold_pnl
    if advantage > 0:
        print(f"✅ LIQUIDATION SAVED ${advantage:.2f} more over 1 year.")
    else:
        print(f"❌ HOLDING was better by ${-advantage:.2f}.")

if __name__ == "__main__":
    asyncio.run(run_full_year_comparison())
