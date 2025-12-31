import asyncio
import logging
import sys
from termcolor import colored

from .config import ARB_THRESHOLD, POLL_INTERVAL, DRY_RUN
from .market_monitor import MarketMonitor
from .arb_detector import check_arb_opportunity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PolymarketArb")

async def main():
    logger.info("Starting Polymarket Arb Bot...")
    logger.info(f"Configuration: Threshold={ARB_THRESHOLD}, Poll_Interval={POLL_INTERVAL}s, Dry_Run={DRY_RUN}")
    
    monitor = MarketMonitor()
    await monitor.start()
    
    try:
        while True:
            logger.info("Fetching top markets...")
            # Fetch more markets to ensure we find a simple Binary one
            markets = await monitor.get_top_markets(limit=50) 
            
            if not markets:
                logger.warning("No markets found. Retrying...")
                await asyncio.sleep(POLL_INTERVAL)
                continue
                
            logger.info(f"Scanning {len(markets)} markets for binary (Yes/No) pairs...")
            
            found_valid_market = False

            for i, market in enumerate(markets):
                clob_token_ids = market.get("clobTokenIds", [])
                
                # Rigid filter for Binary markets (only 2 tokens)
                if len(clob_token_ids) != 2:
                    if i < 5: logger.info(f"[DEBUG Skip] {market.get('question')} | TokenIDs Len: {len(clob_token_ids)}")
                    continue 
                
                try:
                    token_yes = clob_token_ids[0]
                    token_no = clob_token_ids[1]
                    
                    # Fetch orderbooks
                    book_yes = await monitor.get_orderbook(token_yes)
                    book_no = await monitor.get_orderbook(token_no)
                    
                    # Get Best Asks
                    asks_yes_list = book_yes.get("asks", [])
                    asks_no_list = book_no.get("asks", [])
                    
                    if not asks_yes_list or not asks_no_list:
                        if i < 10: 
                             logger.info(f"[DEBUG Skip] {market.get('question')} | No Liquidity (YES: {len(asks_yes_list)}, NO: {len(asks_no_list)})")
                             if not asks_yes_list: logger.info(f"   -> YES Book: {book_yes}")
                        continue

                    ask_yes = float(asks_yes_list[0][0])
                    ask_no = float(asks_no_list[0][0])
                    
                    is_arb, total_cost = check_arb_opportunity(ask_yes, ask_no, ARB_THRESHOLD)
                    
                    question = market.get("question", "Unknown Question")
                    
                    log_msg = f"Market: {question[:60]}... | YES: {ask_yes} | NO: {ask_no} | Sum: {total_cost:.4f}"
                    
                    # Log the first valid one we see to prove it works
                    if not found_valid_market:
                        logger.info(f"[VERIFIED] {log_msg}")
                        found_valid_market = True

                    if is_arb:
                        logger.info(colored(f"[ARB DETECTED] {log_msg} | PROFIT: {(1.0-total_cost)*100:.2f}%", "green", attrs=["bold"]))
                                
                except Exception as ex:
                    if i < 5: logger.info(f"[DEBUG Error] {market.get('question')} | {ex}")
                    continue
            
            if not found_valid_market:
                 logger.info("No active binary markets found with liquidity in this batch.")

            # For verification test, slow down the loop significantly
            logger.info("Debug run complete (sleeping 60s)...")
            await asyncio.sleep(60.0)
            
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
    finally:
        await monitor.stop()

if __name__ == "__main__":
    try:
        # Check python version for asyncio.run support
        if sys.version_info >= (3, 7):
            asyncio.run(main())
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
