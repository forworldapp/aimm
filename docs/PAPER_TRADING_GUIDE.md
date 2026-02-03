# Paper Trading Guide - AIMM v6.0

> 실행 전 필수 읽기

## Quick Start

```bash
# 1. 환경 설정
cd c:\Antigravity\resources\app\scratch\aimm

# 2. Paper Trading 모드 확인 (config.yaml)
# exchange.paper_trading: true 인지 확인

# 3. 봇 실행
python main.py
```

---

## Pre-flight Checklist

### 1. Config 확인 (`config.yaml`)

```yaml
# ✅ Paper Trading 모드
exchange:
  paper_trading: true
  testnet: true  # GRVT Testnet

# ✅ Risk 설정
risk:
  max_position_usd: 5000
  max_loss_usd: 200  # Circuit Breaker

# ✅ 활성화된 ML 모듈
order_flow_analysis:
  enabled: true

funding_rate_arbitrage:
  enabled: true   # +$890/year

execution_algo:
  enabled: true   # +$94/year

# ❌ 비활성화된 모듈 (건드리지 마세요)
microstructure_signals:
  enabled: false

cross_asset_hedge:
  enabled: false

rl_agent:
  enabled: false
```

### 2. 데이터 확인

```bash
# 1년 데이터 존재 확인
ls data/btcusdt_1m_1year.csv

# 모델 파일 확인
ls data/*.pkl
ls models/*.zip
```

### 3. 대시보드 실행

```bash
# 별도 터미널에서
streamlit run dashboard.py
```

---

## 모니터링 항목

| Metric | 정상 범위 | 경고 |
|--------|----------|------|
| PnL | -$50 ~ +$100/day | < -$100 |
| Position | < $3000 | > $4000 |
| Trades | 50-200/hour | < 10 or > 500 |
| Latency | < 500ms | > 2000ms |
| Funding Rate | -0.1% ~ +0.1% | > ±0.3% |

---

## 주요 로그 확인

```bash
# 실시간 로그
tail -f logs/bot.log

# 에러만 필터
grep -i "error\|warning\|circuit" logs/bot.log
```

**정상 로그 예시:**
```
[INFO] MarketMaker cycle started
[INFO] 📊 Order Flow: BALANCED | Spread=1.0x | Size=1.0x
[INFO] 💰 Funding Rate: +0.010% (8h) | Long bias | Bid×0.9 Ask×1.1
[INFO] Placed orders: BID $49,850 x 0.004 | ASK $49,950 x 0.004
```

**경고 로그 예시:**
```
[WARNING] 🛑 Circuit Breaker: Loss $180 (limit $200)
[WARNING] Funding Rate Freeze: 25 min to settlement
```

---

## 비상 정지

### 자동 정지 (Circuit Breaker)
- Loss > $200 → 자동 포지션 청산

### 수동 정지
```bash
# 터미널에서 Ctrl+C

# 또는 별도 터미널에서
python scripts/emergency_close.py
```

---

## 예상 성과 (1년 백테스트 기준)

| Module | 기여 |
|--------|------|
| Base Market Making | ~$1,000 |
| Funding Rate (+) | +$890 |
| TWAP Execution (+) | +$94 |
| Order Flow (위험 감소) | - |
| **Total** | **~$1,984** |

⚠️ 실제 결과는 시장 상황에 따라 다를 수 있습니다.

---

## Troubleshooting

| 문제 | 해결 |
|------|------|
| API 연결 실패 | `config.yaml`의 API 키 확인 |
| 모델 로드 실패 | `data/*.pkl` 파일 존재 확인 |
| 주문 거부 | 잔고 및 레버리지 설정 확인 |
| 대시보드 빈칸 | `streamlit run dashboard.py` 재실행 |

---

## 다음 단계

1. **24시간 Paper Trading** 모니터링
2. **결과 분석** - PnL, 체결률, 슬리피지
3. **파라미터 튜닝** 필요시
4. **Live Trading 전환** - `paper_trading: false`
