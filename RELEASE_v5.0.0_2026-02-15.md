# AIMM Bot v5.0.0 — Live Trading Release (2026-02-15)

## 🎯 Overview
AIMM Bot이 Paper Trading에서 **Live Trading (GRVT Mainnet)** 으로 전환되었습니다.
실제 자금으로 BTC_USDT_Perp 시장조성 전략을 실행합니다.

---

## 📋 Change Log

### 🔴 [MAJOR] Live Trading Activation
- `config.yaml`: `mode: paper` → `mode: live`
- GRVT Mainnet (Prod) 환경으로 전환
- 실제 자금 $14,063 USDT 계좌 연결

### 🔐 [SECURITY] API Credentials Management
- `.env` 파일에 GRVT API 키, Private Key 안전하게 저장
- `.gitignore`에 `.env` 추가 (기존 적용됨)
- `GRVT_TRADING_ACCOUNT_ID`: 숫자 형식 sub_account_id 발견 및 적용
  - 이전: `35IB75FKEUbGlw5MDW1azb05Iru` (main account ID — 오류 원인)
  - 현재: `8785726619222876` (trading sub_account_id — 정상)

### 🔧 [FIX] Sub Account ID Resolution
- **문제**: GRVT SDK의 EIP712 서명에서 `subAccountID`를 `uint64`로 요구
- **원인**: GRVT 대시보드의 ID는 main account ID (알파벳 포함), SDK는 숫자 sub-account ID 필요
- **해결**: `get_sub_accounts` API 호출로 올바른 숫자 ID `8785726619222876` 발견
- **파일**: `core/grvt_exchange.py`, `.env`

### 📊 [FEATURE] Bot-Only PnL Isolation
수동매매 포지션과 봇 자동매매 손익이 혼합되지 않도록 분리:

- **`_bot_order_ids` 추적**: `place_limit_order()` 성공 시 order_id 저장
- **`fetch_and_save_trades()` 필터링**: `order_id in _bot_order_ids` 조건으로 봇 주문 체결만 기록
- **FIFO Grid Profit**: 봇 거래만으로 정확한 그리드 수익 계산
- **Paper Trading 데이터 분리**: 이전 CSV → `_paper_backup` 파일로 이동

### 🛡️ [FEATURE] Emergency Stop (이전 커밋)
- 대시보드 Emergency Stop 버튼 추가
- Telegram 알림 연동

---

## 📁 Modified Files

| File | Change |
|------|--------|
| `config.yaml` | `mode: live` |
| `core/grvt_exchange.py` | Sub account ID fix, `_bot_order_ids` tracking, trade filtering |
| `strategies/market_maker.py` | Enable `fetch_and_save_trades()` call |
| `.env` | `GRVT_TRADING_ACCOUNT_ID=8785726619222876` (gitignored) |
| `tools/test_auth.py` | GRVT auth debugging utility (NEW) |

## 📊 Trading Parameters (Unchanged)
- **Symbol**: BTC_USDT_Perp
- **Grid**: 7 levels × 0.15% spacing
- **Order Size**: 0.002 BTC per level
- **Max Position**: $5,000
- **ML Model**: Direction prediction + Volatility regime

## ⚠️ Risk Notes
- 라이브 모드에서는 실제 자금이 사용됩니다
- Emergency Stop: `python tools/emergency_stop.py` 또는 대시보드 버튼
- 최대 포지션 $5,000 제한 적용
- Circuit Breaker: 연속 손실 시 자동 정지

---

## 🏗️ Architecture
```
main.py
├── core/grvt_exchange.py    ← Live GRVT SDK 연결
│   ├── _bot_order_ids       ← 봇 주문 ID 추적
│   ├── place_limit_order()  ← 주문 배치 + ID 저장
│   └── fetch_and_save_trades() ← 봇 체결만 기록
├── strategies/market_maker.py ← MM 전략 실행
├── dashboard.py             ← Streamlit 실시간 대시보드
└── tools/emergency_stop.py  ← 긴급 정지
```
