# v4.0 LightGBM Direction Predictor - Implementation Plan

## 🎯 목표

HMM 레짐 감지에 LightGBM 가격 방향 예측을 추가하여 그리드 주문 편향을 최적화합니다.

---

## 📊 현재 시스템 (v3.8.1) vs 개선 시스템 (v4.0)

### v3.8.1 (현재)
```
Binance Data → HMM → Regime (low_vol, high_vol, trend_up, trend_down)
                 ↓
            Parameter Blending (γ, κ, grid_layers, order_size)
                 ↓
            Grid Market Maker
```

### v4.0 (개선)
```
Binance Data → HMM → Regime Detection
     ↓
LightGBM → Direction Prediction (UP/DOWN/NEUTRAL, 확률)
     ↓
Combined Signal → Enhanced Parameter Blending
     ↓
Grid Market Maker (방향 편향 주문)
     ↓
Performance Monitor → Drift Detection → Auto Retrain (필요시)
```

---

## 🔧 영향받는 파라미터

### 1. 주문 편향 (Order Skew) - 신규

| 파라미터 | 현재 | v4.0 |
|---------|------|------|
| bid_skew | 재고 기반만 | 재고 + LightGBM 방향 |
| ask_skew | 재고 기반만 | 재고 + LightGBM 방향 |

**예시:**
- LightGBM이 "UP 65%" 예측 → 매수 주문 더 공격적 (가격↑, 수량↑)
- LightGBM이 "DOWN 70%" 예측 → 매도 주문 더 공격적

### 2. 그리드 레이어 배치 - 개선

| 파라미터 | 현재 | v4.0 |
|---------|------|------|
| grid_layers | HMM 레짐별 고정 | 방향 확률로 비대칭 배치 |
| layer_spacing | 균등 간격 | 방향 쪽 더 촘촘하게 |

### 3. 주문 크기 (Order Size) - 개선

| 파라미터 | 현재 | v4.0 |
|---------|------|------|
| bid_size_usd | HMM 블렌딩 | HMM + 방향 신뢰도 가중 |
| ask_size_usd | HMM 블렌딩 | HMM + 방향 신뢰도 가중 |

### 4. 스프레드 조정 - 개선

| 파라미터 | 현재 | v4.0 |
|---------|------|------|
| spread_pct | 변동성 기반 | 변동성 + 방향 확신도 |

---

## 🧠 레짐 로직 변화

### HMM + LightGBM 결합 매트릭스

| HMM 레짐 | LightGBM 예측 | 최종 전략 |
|---------|--------------|----------|
| low_vol | UP 60%+ | 약한 롱 편향 |
| low_vol | DOWN 60%+ | 약한 숏 편향 |
| low_vol | NEUTRAL | 순수 그리드 |
| high_vol | UP 70%+ | 강한 롱 편향 + 넓은 스프레드 |
| high_vol | DOWN 70%+ | 강한 숏 편향 + 넓은 스프레드 |
| trend_up | UP 60%+ | 최대 롱 공격 |
| trend_up | DOWN 60%+ | 충돌 → HMM 우선 |
| trend_down | DOWN 60%+ | 최대 숏 공격 |
| trend_down | UP 60%+ | 충돌 → HMM 우선 |

### 충돌 해결 로직

```python
def resolve_hmm_lgb_conflict(hmm_regime: str, lgb_prediction: str, lgb_confidence: float) -> tuple:
    """
    HMM과 LightGBM 신호 충돌 시 해결 로직
    
    Returns:
        (final_direction, confidence_multiplier)
    """
    conflict_matrix = {
        ('trend_up', 'DOWN'): {
            'high_conf': ('UP', 0.3),    # 70%+ → HMM 따르되 약하게
            'med_conf': ('UP', 0.5),     # 60-70%
            'low_conf': ('UP', 0.7),     # 55-60%
        },
        ('trend_down', 'UP'): {
            'high_conf': ('DOWN', 0.3),
            'med_conf': ('DOWN', 0.5),
            'low_conf': ('DOWN', 0.7),
        },
        ('low_vol', 'UP'): {
            'high_conf': ('UP', 0.5),    # 브레이크아웃 가능성
            'med_conf': ('UP', 0.3),
            'low_conf': ('NEUTRAL', 1.0),
        },
        ('low_vol', 'DOWN'): {
            'high_conf': ('DOWN', 0.5),
            'med_conf': ('DOWN', 0.3),
            'low_conf': ('NEUTRAL', 1.0),
        },
        ('high_vol', 'UP'): {
            'high_conf': ('UP', 0.4),
            'med_conf': ('NEUTRAL', 0.5),
            'low_conf': ('NEUTRAL', 1.0),
        },
        ('high_vol', 'DOWN'): {
            'high_conf': ('DOWN', 0.4),
            'med_conf': ('NEUTRAL', 0.5),
            'low_conf': ('NEUTRAL', 1.0),
        },
    }
    
    if lgb_confidence >= 0.70:
        conf_level = 'high_conf'
    elif lgb_confidence >= 0.60:
        conf_level = 'med_conf'
    else:
        conf_level = 'low_conf'
    
    key = (hmm_regime, lgb_prediction)
    if key in conflict_matrix:
        return conflict_matrix[key][conf_level]
    
    return (lgb_prediction, lgb_confidence)
```

---

## 📈 LightGBM 모델 설계

### 입력 피처 (~45개)

```python
features = {
    'price': [
        'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
        'volatility_20', 'volatility_60',
        'high_low_range', 'close_to_vwap',
    ],
    'technical': [
        'rsi_14', 'rsi_7', 'bb_pct', 'bb_width',
        'macd', 'macd_signal', 'macd_histogram',
        'ema_cross_9_21', 'atr_14', 'adx_14',
    ],
    'microstructure': [
        'orderbook_imbalance', 'spread_bps',
        'mid_price_velocity', 'depth_imbalance_l5',
    ],
    'trade_flow': [
        'buy_sell_ratio', 'large_trade_ratio',
        'cvd_1m', 'cvd_5m', 'volume_ratio', 'volume_ma_ratio',
    ],
    'derivatives': [
        'funding_rate', 'funding_rate_ma_8h',
        'oi_change_1h', 'oi_change_24h', 'long_short_ratio',
    ],
    'cross_market': [
        'btc_correlation_15m', 'btc_returns_5m', 'eth_btc_ratio_change',
    ],
    'temporal': [
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_asia_session', 'is_europe_session', 'is_us_session',
        'minutes_to_funding',
    ],
    'regime': [
        'regime_low_vol', 'regime_high_vol',
        'regime_trend_up', 'regime_trend_down',
        'regime_duration',
    ],
}
```

### 출력 (Target)

```python
target_map = {
    0: 'DOWN',    # < -0.05%
    1: 'NEUTRAL', # -0.05% ~ +0.05%
    2: 'UP'       # > +0.05%
}
```

---

## 🏋️ 모델 학습 프로세스

### Walk-Forward Validation

```python
class WalkForwardValidator:
    def __init__(
        self,
        n_splits: int = 5,
        train_period_days: int = 60,
        test_period_days: int = 7,
        gap_days: int = 1
    ):
        self.n_splits = n_splits
        self.train_period = train_period_days * 1440
        self.test_period = test_period_days * 1440
        self.gap = gap_days * 1440
```

### 하이퍼파라미터 튜닝 (Optuna)

```python
params = {
    'num_leaves': 20-100,
    'max_depth': 3-12,
    'learning_rate': 0.01-0.1,
    'n_estimators': 100-500,
    'min_data_in_leaf': 50-200,
    'feature_fraction': 0.6-0.9,
    'bagging_fraction': 0.6-0.9,
    'lambda_l1': 1e-8-10.0,
    'lambda_l2': 1e-8-10.0,
}
```

---

## ⚙️ config.yaml 설정

```yaml
lightgbm_predictor:
  enabled: true
  model_path: data/direction_model_lgb.pkl
  prediction_horizon: 1
  confidence_threshold: 0.55
  neutral_zone: [0.45, 0.55]
  skew_multiplier: 1.0
  size_adjustment: true
  layer_asymmetry: true

lightgbm_training:
  hyperparameter_tuning:
    enabled: true
    method: optuna
    n_trials: 100
  validation:
    method: walk_forward
    n_splits: 5
    train_period_days: 60
    test_period_days: 7
    gap_days: 1
  regularization:
    early_stopping_rounds: 50
    min_data_in_leaf: 100
    feature_fraction: 0.8

lightgbm_operations:
  retraining:
    frequency: weekly
    trigger_conditions:
      accuracy_drop_threshold: 0.03
      consecutive_wrong_predictions: 15
      psi_threshold: 0.25
  monitoring:
    metrics: [accuracy, precision, recall, f1, profit_contribution]
    window_size: 1000
    alert_thresholds:
      accuracy_min: 0.48
      f1_min: 0.45
  drift_detection:
    enabled: true
    method: PSI
    threshold: 0.25
    action: reduce_weight_and_alert

lightgbm_risk_management:
  max_skew_limits:
    size_multiplier_max: 1.5
    size_multiplier_min: 0.5
    layer_asymmetry_max: [7, 3]
    spread_adjustment_max_pct: 20
  fallback:
    on_model_error: use_hmm_only
    on_low_confidence: use_neutral
    on_drift_detected: reduce_weight_50pct
  loss_limits:
    max_directional_loss_usd: 500
    daily_lgb_loss_limit_usd: 1000
    weekly_lgb_loss_limit_usd: 3000
  consecutive_miss_handling:
    threshold: 5
    action: reduce_skew_50pct
    threshold_severe: 10
    action_severe: disable_lgb_1h
    cooldown_minutes: 30
  extreme_market_conditions:
    volatility_spike_threshold: 3.0
    action: use_hmm_only

backtest_requirements:
  period:
    in_sample: "2023-01-01 to 2024-06-30"
    out_of_sample: "2024-07-01 to 2024-12-31"
  costs:
    maker_fee_bps: 2
    taker_fee_bps: 5
    slippage_model: volume_based
  acceptance_criteria:
    sharpe_ratio_min: 1.5
    max_drawdown_max: 0.10
    profit_factor_min: 1.3
    win_rate_min: 0.50
    vs_baseline:
      sharpe_improvement_min: 0.3
      return_improvement_min: 0.05

deployment:
  phase_1_paper:
    duration_days: 14
    capital: 0
    lgb_weight: 1.0
    success_criteria:
      sharpe_ratio_min: 1.5
  phase_2_small:
    duration_days: 14
    capital_usd: 1000
    lgb_weight: 0.5
    success_criteria:
      total_pnl_min: 0
  phase_3_medium:
    duration_days: 30
    capital_usd: 5000
    lgb_weight: 0.75
    success_criteria:
      sharpe_ratio_min: 1.3
  phase_4_full:
    duration_days: 30
    capital_usd: 10000
    lgb_weight: 1.0
  phase_5_scale:
    capital: full_allocation
```

---

## 📁 파일 구조

```
ml/
├── lightgbm_predictor.py       # [NEW]
├── train_lightgbm.py           # [NEW]
├── feature_engineering.py      # [NEW]
├── walk_forward_validator.py   # [NEW]
├── drift_detector.py           # [NEW]
└── hmm_regime_detector.py      # [EXISTING]

monitoring/
├── lgb_performance_monitor.py  # [NEW]
├── alert_manager.py            # [NEW]
└── dashboard.py                # [NEW]

data/
├── regime_model_hmm.pkl        # [EXISTING]
├── direction_model_lgb.pkl     # [NEW]
├── feature_scaler.pkl          # [NEW]
└── training_metadata.json      # [NEW]

strategies/market_maker.py      # [MODIFY]
config.yaml                     # [MODIFY]
```

---

## ✅ 구현 단계

### Phase 0: 준비 (1일)
- [ ] 의존성 확인 (lightgbm, optuna, scikit-learn)
- [ ] 데이터 가용성 확인 (최소 1년 1분봉)

### Phase 1: 피처 파이프라인 (2-3일)
- [ ] `ml/feature_engineering.py` 생성
- [ ] 피처 품질 검증

### Phase 2: 모델 구현 (2-3일)
- [ ] `ml/lightgbm_predictor.py` 생성
- [ ] `ml/walk_forward_validator.py` 생성
- [ ] `ml/train_lightgbm.py` 생성

### Phase 3: 하이퍼파라미터 튜닝 (1-2일)
- [ ] Optuna 튜닝 실행 (100 trials)

### Phase 4: 모델 학습 및 검증 (1일)
- [ ] 최종 모델 학습
- [ ] Walk-forward 검증

### Phase 5: 백테스트 (2-3일)
- [ ] In-sample / Out-of-sample 백테스트
- [ ] Acceptance criteria 검증

### Phase 6: 통합 (2-3일)
- [ ] `strategies/market_maker.py` 수정
- [ ] 리스크 관리 로직 구현

### Phase 7: 모니터링 (1-2일)
- [ ] 성능 모니터, Drift 감지 구현

### Phase 8-10: 배포 (60일+)
- [ ] Phase 1-5 순차 배포
- [ ] 태그 `v4.0.0-lightgbm` 생성

---

## 🔍 체크리스트

### 학습 전
- [ ] 미래 정보 누수 없음
- [ ] 시계열 순서 유지
- [ ] 클래스 불균형 처리

### 배포 전
- [ ] Out-of-sample 성능 충족
- [ ] 리스크 관리 테스트 완료
- [ ] 폴백 로직 테스트 완료

---

*Version: v4.0 Final*
*Date: 2026-01-27*
