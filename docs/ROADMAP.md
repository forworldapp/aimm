# GRVT Bot Development Roadmap

## Phase 1: 안정화 (Completed ✅)
- [x] RiskManager 실제 적용
- [x] Drawdown 자동 정지 구현
- [x] 예외 처리 강화 (GrvtExchange Retry Logic applied)
- [x] 로깅 시스템 개선 (RotatingFileHandler configured)

## Phase 2: Intelligence & Optimization (v1.2 - v1.3)
- [x] **Adaptive Trend Strategy** (v1.2)
    - [x] Auto-detect Ranging vs Trending using MA Divergence.
    - [x] Dynamic Skew Adjustment.
- [x] **Advanced Technical Filters** (v1.3)
    - [x] Implement ADX, ATR, Choppiness Index Filters.
    - [x] Create 'Combo' Filter (ADX + ATR) for high-probability entries.
    - [x] Integrate Candle Data (OHLC) processing.
- [x] **Dashboard V2** (v1.3)
    - [x] Selectable Strategy Mode (Combo, ADX, MA, etc).
    - [x] Real-time Regime Status Display.

## Phase 3: Risk & Advanced Features (v1.4+)
- [ ] **Dynamic Parameters**
    - [ ] Auto-adjust Spread based on ATR/Volatility.
    - [ ] [TODO] restore ADX period to 14.mbol architecture)
- [ ] 백테스트 프레임워크 (Backtesting engine)
- [ ] Live 모드 테스트 (Small capital test)

## Phase 4: 고급 기능 (Future)
- [ ] 머신러닝 기반 스프레드 최적화 (ML Spread Optimization)
- [ ] 실시간 펀딩비 연동 (Funding Arbitrage)
- [ ] Cross-Exchange Arbitrage

---

## 📋 우선순위 액션 아이템
| 순위 | 작업 | 진행상태 |
|:---:|:---|:---:|
| 1 | RiskManager 연동 | ✅ Done |
| 2 | Drawdown 체크 구현 | ✅ Done |
| 3 | 스마트 Cancel & Replace | ✅ Done |
| 4 | Grid 레이어 주문 | ✅ Done |
| 5 | 동적 트렌드 Skew | ✅ Done |
| 6 | 동적 스프레드 로직 | Pending |
