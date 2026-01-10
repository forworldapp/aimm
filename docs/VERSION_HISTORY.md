# GRVT Market Maker Bot - Version History

## Tag List

| Version | Description |
|---------|-------------|
| **v2.6.0** | ML Regime Detection: K-Means 기반 시장 레짐 감지 및 동적 γ/κ 파라미터 조정 |
| **v2.5.0** | Avellaneda-Stoikov Model: 학술 논문 기반 동적 스프레드/예약가격 구현 |
| **v2.4.0** | Signal Boost Strategy: 시그널 감지 시 공격적 추가 주문, 양방향 그리드 유지 |
| **v2.3.2** | Fix: Dashboard refresh (2s), Paper I/O Stability, Tolerance 0.1% |
| **v2.3.1** | Hotfix: Order Size $200 (Min Notional), Max Loss UI |
| **v2.3.0** | Risk Management Update (FIFO Grid Profit, Max Position, Circuit Breaker) |
| **v2.2** | Live Trading 지원, GRVT API 호환성 수정 |
| v2.0.0-rc1 | Production Ready Build |
| v1.9.4 | Fix: grid_profit 초기화 |
| v1.9.3 | Fix: Entry price 가중평균 계산 |
| v1.9.2 | FIFO Grid Profit 정확도 |
| v1.9.1 | Smart Order Management |
| v1.9.0 | Grid Profit 추적 |
| v1.8.2 | 기존 포지션 Profit-taking |
| v1.8.1 | Neutral/Waiting 모드 주문 유지 |
| v1.8.0 | Signal-Only Trading Mode |
| v1.7.3 | DCA Throttle 제거, 인벤토리 스큐 강화 |
| v1.7.2 | Loss State Adjustment |
| v1.7.1 | VaR 기반 인벤토리 |
| v1.6.1 | 선택적 취소, Break-even 보존 |
| v1.6.0 | Grid Profit 추적 및 표시 |
| v1.5.2 | ATR 기반 동적 그리드 |
| v1.5.0-circuitbreaker | Statistical Circuit Breaker |
| v1.4.6 | DCA Throttle (최소 가격 거리) |
| v1.4.5 | Inventory Relief Logic |
| v1.4.4 | BB RSI Neutral Trading Strategy |
| v1.4.4-dashboard | Dashboard 추가 |
| v1.4.3 | Stable Release |
| v1.3 | 초기 전략 버전 |
| v1.2 | 기본 구조 |
| v1.0.0 | 초기 릴리스 |

---

## Branch Structure

```
main (현재 v1.8.2 기반)
    │
    ├── v1.4.4 ← v2.2 (Live Trading)
    │       │
    │       └── ... v1.9.x, v2.0.0-rc1
    │
    └── v1.5.x ~ v1.8.x (Paper Trading 최적화)
```

---

## v2.6.0 변경사항 (2026-01-10)

### 새로운 기능
- ✅ **ML 레짐 감지**: K-Means 클러스터링으로 4개 시장 레짐 자동 분류
  - 🟢 `low_vol`: 낮은 변동성 → γ=1.5, κ=2000 (좁은 스프레드)
  - 🔵 `trend_up`: 상승 추세 → γ=0.5, κ=500 (넓은 스프레드, 매도 선호)
  - 🔴 `trend_down`: 하락 추세 → γ=0.5, κ=500 (넓은 스프레드, 매수 선호)
  - 🟠 `high_vol`: 높은 변동성 → γ=0.3, κ=200 (매우 넓은 스프레드)
- ✅ **동적 파라미터 전환**: 감지된 레짐에 따라 A&S γ/κ 자동 조정
- ✅ **대시보드 표시**: ML 레짐 실시간 표시 (이모지 컬러 코드)

### 새 파일
- `ml/regime_detector.py`: K-Means 레짐 감지 모델
- `data/regime_model.pkl`: 훈련된 모델 저장
- `data/btc_hourly_1000.csv`: Binance BTC 1시간 캔들 학습 데이터

### 설정
```yaml
strategy:
  ml_regime_enabled: true  # ML 레짐 감지 활성화
```

---

## v2.5.0 변경사항 (2026-01-09)

### 새로운 기능
- ✅ **Avellaneda-Stoikov 모델**: 학술 논문 기반 최적 마켓메이킹
  - 예약가격: `r = s - q × γ × σ² × (T-t)`
  - 최적 스프레드: `δ = γ × σ² × (T-t) + (2/γ) × ln(1 + γ/κ)`
- ✅ **변동성 기반 동적 스프레드**: σ(변동성) 증가 시 스프레드 자동 확대
- ✅ **대시보드 A&S 메트릭**: 예약가격, 스프레드, 변동성, γ/κ 실시간 표시

### 설정
```yaml
strategy:
  avellaneda_stoikov:
    enabled: true
    gamma: 1.0    # 위험 회피 계수
    kappa: 1000   # 유동성 계수
    min_spread: 0.001
    max_spread: 0.01
```

---

## v2.2 변경사항 (2026-01-08)

### 새로운 기능
- ✅ **Live Trading 지원**: GRVT 실거래 환경 연결
- ✅ **GRVT API 호환성**: `order_id`, `legs[]`, `instrument` 키 사용
- ✅ **미실현 손익 반영**: `get_account_summary()` 사용

### 수정사항
- `place_limit_order`: `order_type='limit'` 파라미터
- `get_position`: GRVT `instrument/size/entry_price` 키
- `cancel_all_orders`: `order_id` 사용
- `Smart Order Management`: 0.5% tolerance 적용

### 새 문서
- `docs/SETUP_GUIDE.md`: Paper/Live 셋업 가이드
- `docs/EXECUTION_GUIDE.md`: 실행 방법 가이드

---

## 머지 충돌 히스토리

**2026-01-08**: v2.1 → main 머지 시 충돌 발생
- 원인: main과 작업 브랜치가 분기됨 (15 vs 7 커밋)
- 해결: v2.2로 새 태그 생성 후 푸시

---

## 권장 사용 버전

| 용도 | 권장 버전 |
|------|----------|
| **Live Trading** | v2.2 |
| **Paper Trading** | v2.2 또는 v2.0.0-rc1 |
| **개발/테스트** | main 또는 v2.2 |
