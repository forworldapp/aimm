# GRVT Bot Development Roadmap

## Phase 1: 안정화 (Completed ✅)
- [x] RiskManager 실제 적용
- [x] Drawdown 자동 정지 구현
- [x] 예외 처리 강화 (GrvtExchange Retry Logic applied)
- [x] 로깅 시스템 개선 (RotatingFileHandler configured)

## Phase 2: 전략 고도화 (In Progress 🏗️)
- [x] Grid 레이어 주문 (Multi-layer Grid implemented)
- [x] 스마트 Cancel & Replace (Only update on significant price change)
- [x] 동적 트렌드 Skew (MA-based Trend Mode & Toggle)
- [x] 평단가 방어 로직 (Entry Anchor Mode for Ranging Markets)
- [ ] 동적 스프레드 (Based on Orderbook Imbalance/Depth)
- [ ] 트렌드 필터 강화 (Integrate RSI, MACD indicators)

## Phase 3: 인프라 (Upcoming)
- [ ] Telegram/Discord 알림
- [ ] 다중 심볼 지원 (Multi-symbol architecture)
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
