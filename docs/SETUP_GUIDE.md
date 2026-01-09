# GRVT Market Maker Bot - Setup Guide

## Version 2.1

---

## 1. Prerequisites

- Python 3.10+
- Git
- GRVT Exchange Account (with API keys)

---

## 2. Clone Repository

```bash
git clone https://github.com/your-repo/grvt_bot.git
cd grvt_bot
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configuration

### 4.1 Environment Variables (`.env`)

Create `.env` file in the project root:

```env
# GRVT API Credentials
GRVT_API_KEY=your_api_key_here
GRVT_PRIVATE_KEY=your_private_key_here
GRVT_TRADING_ACCOUNT_ID=your_trading_account_id

# Environment (testnet or prod)
GRVT_ENV=prod
```

### 4.2 Trading Mode (`config.yaml`)

**For Paper Trading:**
```yaml
exchange:
  env: testnet
  mode: paper
  symbol: BTC_USDT_Perp
```

**For Live Trading:**
```yaml
exchange:
  env: prod
  mode: live
  symbol: BTC_USDT_Perp
```

---

## 5. Running the Bot

### Start Bot
```bash
python main.py
```

### Start Dashboard (separate terminal)
```bash
python -m streamlit run dashboard.py --server.port 8503
```

### Access Dashboard
- **Local**: http://localhost:8503

---

## 6. Key Configuration Options

| Config | Description | Default |
|--------|-------------|---------|
| `risk.max_drawdown_pct` | Max drawdown before stop | 0.05 (5%) |
| `risk.max_position_usd` | Max position size | $500 |
| `strategy.order_size_usd` | Order size per layer | $100 |
| `strategy.grid_layers` | Number of grid layers | 5 |
| `strategy.spread_pct` | Base spread | 0.0025 (0.25%) |

---

## 7. Stopping the Bot

```bash
# Windows PowerShell
Get-Process python | Stop-Process -Force

# Or press Ctrl+C in the terminal
```

---

## 8. Troubleshooting

### Authentication Error (401)
- Verify API keys in `.env`
- Check `trading_account_id` is correct
- Ensure API keys have trading permissions

### Order Size Too Granular
- GRVT requires 0.001 BTC minimum lot size
- Increase `order_size_usd` if needed

### Rate Limit
- Bot has built-in retry logic
- If persistent, increase `refresh_interval`

---

## 9. Paper vs Live Differences

| Aspect | Paper | Live |
|--------|-------|------|
| Exchange Class | `PaperGrvtExchange` | `GrvtExchange` |
| Fills | Simulated | Real |
| Fees | Simulated | Real |
| Risk | None | Real Money |

---

## Version History

- **v2.1**: Live trading support, GRVT API fixes, Smart Order Management
- **v2.0**: Bollinger Band strategy, Dynamic spread, Dashboard
- **v1.0**: Initial paper trading implementation
