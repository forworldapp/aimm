# ML 기반 마켓메이킹 개선 로드맵

## 개요
현재 A&S 모델 기반 전략을 ML로 강화하여 동적 파라미터 최적화, 시장 레짐 감지, 자동 튜닝을 구현하는 단계별 계획

---

## Phase 1: 데이터 수집 인프라 (1-2주)

### 목표
ML 모델 학습을 위한 고품질 데이터 축적

### 구현 사항
1. **거래 로그 확장** (`trade_history.csv`)
   - 추가 필드: `spread_at_trade`, `volatility`, `regime`, `orderbook_imbalance`
   
2. **시장 스냅샷 저장** (새 파일)
   - 1초마다: mid_price, spread, position, volatility, orderbook_depth
   - 저장 위치: `data/market_snapshots_{date}.csv`

3. **레이블링 자동화**
   - 거래별 수익성 태깅 (profitable/unprofitable)
   - 레짐 사후 태깅 (trending/ranging/volatile)

### 성공 기준
- 최소 1,000건 거래 + 100,000 시장 스냅샷 축적

---

## Phase 2: 시장 레짐 감지 (2-3주)

### 목표
실시간 시장 상태 분류 (Trending / Mean-Reverting / High-Vol / Low-Vol)

### 접근법
**Option A: Hidden Markov Model (HMM)**
```python
from hmmlearn import GaussianHMM
# States: 4 (Trend-Up, Trend-Down, Range, Volatile)
# Observations: [returns, volatility, volume_ratio]
```

**Option B: K-Means 클러스터링**
```python
features = [volatility, atr_ratio, price_momentum, volume]
kmeans.fit(features)  # 4 clusters
```

### 레짐별 전략 프리셋
| 레짐 | γ (위험회피) | κ (유동성) | 스프레드 |
|------|-------------|-----------|---------|
| Trending | 0.5 | 500 | 넓음 |
| Ranging | 1.0 | 1000 | 좁음 |
| High-Vol | 0.3 | 200 | 매우 넓음 |
| Low-Vol | 1.5 | 2000 | 매우 좁음 |

### 성공 기준
- 레짐 감지 정확도 70%+ (백테스트 검증)
- 레짐별 수익성 차이 확인

---

## Phase 3: 파라미터 자동 튜닝 (2-3주)

### 목표
γ, κ, 스프레드 범위를 성과 기반으로 자동 조정

### 접근법
**베이지안 최적화 (Optuna)**
```python
import optuna

def objective(trial):
    gamma = trial.suggest_float('gamma', 0.1, 2.0)
    kappa = trial.suggest_int('kappa', 100, 5000)
    
    # 백테스트 실행
    pnl = backtest(gamma, kappa)
    return pnl

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 온라인 러닝 (선택)
- 매일 종료 시점에 PnL 평가
- 좋으면 파라미터 유지, 나쁘면 Exploration

### 성공 기준
- 수동 튜닝 대비 PnL 10%+ 개선
- 파라미터 수렴 확인

---

## Phase 4: 강화학습 스프레드 최적화 (4-6주)

### 목표
실시간 시장 상황에 따른 최적 스프레드 결정

### 모델 설계
**State (관측)**
```python
state = [
    normalized_position,     # -1 ~ 1
    volatility_z_score,      # 표준화된 변동성
    spread_percentile,       # 현재 스프레드 백분위
    orderbook_imbalance,     # bid/ask 불균형
    recent_fill_rate,        # 최근 체결률
    pnl_drawdown,            # 손익 하락폭
]
```

**Action (행동)**
```python
actions = [
    "spread_tighten_small",   # -0.01%
    "spread_keep",            # 유지
    "spread_widen_small",     # +0.01%
    "spread_widen_large",     # +0.05%
]
```

**Reward (보상)**
```python
reward = filled_profit - inventory_penalty - spread_cost
# filled_profit: 체결 시 수익
# inventory_penalty: 과도한 포지션 페널티
# spread_cost: 너무 넓은 스프레드 페널티
```

### 학습 방법
- **알고리즘**: PPO (Proximal Policy Optimization) 또는 DQN
- **환경**: Paper Trading 시뮬레이터 활용
- **학습 데이터**: Phase 1에서 수집한 시장 스냅샷

### 성공 기준
- 백테스트 Sharpe Ratio 1.0+
- 라이브 Paper Trading에서 기존 대비 개선

---

## Phase 5: 프로덕션 배포 (2주)

### 안전장치
1. **Shadow Mode**: ML 결정을 로깅만 하고 기존 로직 사용
2. **A/B Testing**: 자금 50%씩 나눠서 비교
3. **Circuit Breaker**: 일 손실 $100 초과 시 ML OFF

### 모니터링
- 대시보드에 ML 결정 표시 (추천 스프레드 vs 실제)
- 레짐 예측 정확도 실시간 추적
- 모델 드리프트 감지

---

## 예상 타임라인

```
Phase 1 (데이터) ████████░░░░░░░░░░░░ 2주
Phase 2 (레짐)  ░░░░░░░░█████████░░░ 3주
Phase 3 (튜닝)  ░░░░░░░░░░░░█████░░░ 2주
Phase 4 (RL)    ░░░░░░░░░░░░░░░█████ 6주
Phase 5 (배포)  ░░░░░░░░░░░░░░░░░░██ 2주
                ──────────────────────
                총 약 15주 (4개월)
```

---

## 리스크 및 완화 전략

| 리스크 | 완화 전략 |
|--------|----------|
| 오버피팅 | Walk-forward validation, Out-of-sample 테스트 |
| 모델 지연 | 추론 시간 <10ms 목표, 캐싱 |
| 데이터 부족 | Paper Trading으로 합성 데이터 생성 |
| 레짐 전환 지연 | 앙상블 모델 (여러 윈도우 크기) |

---

## 다음 단계

1. **Phase 1 시작**: `trade_history.csv` 확장 및 시장 스냅샷 저장 구현
2. **데이터 축적**: 2주간 Paper Trading으로 데이터 모으기
3. **Phase 2 진입**: 충분한 데이터 후 레짐 감지 모델 개발

---

*작성일: 2026-01-09*
*버전: v2.5.0 기준*
