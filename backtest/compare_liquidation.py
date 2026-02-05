import sys
import os
import pandas as pd
import asyncio
import logging

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

# Configure Logging
logging.getLogger("MarketMaker").setLevel(logging.INFO)
logging.getLogger("MockExchange").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

class LiquidationMarketMaker(MarketMaker):
    def __init__(self, exchange, liquidate_on_trigger=False):
        # 1. Force Disable ML in Config BEFORE init if possible, 
        # but MarketMaker init reloads config.
        # So we let it init, then disable.
        super().__init__(exchange)
        
        self.liquidate_on_trigger = liquidate_on_trigger
        
        # 2. Force Disable ML components
        self.ml_regime_enabled = False
        self.regime_detector = None
        self.strategy_v4 = None # Disable v4 logic
        self.adaptive_tuner = None
        
        self.logger.info("⚡ LiquidationMarketMaker: ML components DISABLED for simulation.")

    async def cycle(self):
        # Debug Trace
        if self.exchange.current_index % 500 == 0:
            print(f"   [Cycle] Step {self.exchange.current_index} PnL={self.inventory*0:.2f} (approx)")

        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook: return

        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            # MockExchange returns lists [price, qty]
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
        except Exception as e:
            return
        
        self._update_history(mid_price)
        self._update_candle(mid_price, 0)
        
        current_pos_qty = position.get('amount', 0.0)
        self.inventory = current_pos_qty
        
        # --- Circuit Breaker Check ---
        unrealized_pnl = position.get('unrealizedPnL', 0.0)
        MAX_LOSS = 200.0 # Hardcoded for safety
        
        if unrealized_pnl < -MAX_LOSS:
            print(f"   🚨 TRIGGERED at Step {self.exchange.current_index}! Loss ${abs(unrealized_pnl):.2f}")
            await self.exchange.cancel_all_orders(self.symbol)
            self.is_active = False # Stop
            
            if self.liquidate_on_trigger:
                print(f"   [Action] LIQUIDATING Position: {current_pos_qty} @ {mid_price}")
                await self.exchange.close_position(self.symbol)
                # Verify zero
                pos_after = await self.exchange.get_position(self.symbol)
                print(f"   [Result] New Pos: {pos_after['amount']}")
            else:
                print(f"   [Action] STOPPING (Position Held: {current_pos_qty} @ {mid_price})")
            
            return

        # Standard Logic
        if self.is_active:
            await super().cycle()

async def run_comparison():
    print("\n" + "="*60)
    print("⚔️  LIQUIDATION vs HOLD Comparison (Oct 2025 Crash)")
    print("="*60)
    
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print("Data missing.")
        return
        
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    # 3000 steps to capture crash + aftermath
    df_sim = df.iloc[:3000]
    
    async def run_mode(liquidation_mode, mode_name):
        print(f"\n🔹 Testing Mode: {mode_name}")
        exchange = MockExchange(df_sim, initial_balance=10000.0)
        strategy = LiquidationMarketMaker(exchange, liquidate_on_trigger=liquidation_mode)
        
        # Override Params
        strategy._load_params = lambda: None # Stop reload
        strategy.max_loss_usd = 200.0
        strategy.max_position_usd = 5000.0
        strategy.base_spread = 0.0005 # Tighten base spread to encourage fills (0.05%)
        strategy.is_active = True
        
        eq_at_trigger = 0.0
        triggered = False
        
        for i in range(len(df_sim)):
            if not exchange.next_tick(): break
            
            # Inject Data for strategy compatibility
            idx = exchange.current_index
            strategy.candles = df_sim.iloc[max(0, idx-60):idx+1][['open','high','low','close','volume']]
            strategy.price_history.append((df_sim.iloc[idx]['best_bid'] + df_sim.iloc[idx]['best_ask'])/2)
            if len(strategy.price_history)>200: strategy.price_history.pop(0)

            # Cycle
            # For HOLD strategy: We maintain 'is_active' based on internal state.
            # But the loop should continue running even if stopped, to track equity deviation.
            
            if strategy.is_active:
                await strategy.cycle()
            else:
                # Triggered and Stopped.
                if not triggered:
                    triggered = True
                    eq_at_trigger = exchange.get_equity()
                
                # Standard Mock update happens in next_tick
                pass
        
        final_equity = exchange.get_equity()
        return {
            "Label": mode_name,
            "Triggered": triggered,
            "TriggerEq": eq_at_trigger,
            "FinalEq": final_equity
        }

    res_hold = await run_mode(False, "HOLD (Cancel Only)")
    res_liq = await run_mode(True, "LIQUIDATE (Close Pos)")
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"{'Mode':<20} | {'Triggered':<10} | {'TriggerEq':<15} | {'FinalEq':<15} | {'PnL after Stop':<15}")
    print("-" * 75)
    
    def print_row(r):
        pnl_post = r['FinalEq'] - r['TriggerEq'] if r['Triggered'] else 0.0
        print(f"{r['Label']:<20} | {str(r['Triggered']):<10} | ${r['TriggerEq']:<14.2f} | ${r['FinalEq']:<14.2f} | ${pnl_post:<14.2f}")
        return pnl_post

    pnl_hold = print_row(res_hold)
    pnl_liq = print_row(res_liq)
    print("-" * 75)

    if pnl_liq > pnl_hold:
        diff = pnl_liq - pnl_hold
        print(f"✅ VERDICT: LIQUIDATION saved ${diff:.2f} more than holding!")
    else:
        diff = pnl_hold - pnl_liq
        print(f"❌ VERDICT: HOLDING was better by ${diff:.2f}.")

if __name__ == "__main__":
    asyncio.run(run_comparison())
