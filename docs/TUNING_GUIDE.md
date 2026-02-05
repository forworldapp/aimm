# Tuning & ML Lifecycle Guide

> 파라미터 튜닝, 모델 재학습, 백테스트 기반 최적화 가이드

---

## 1. 실시간 주기

| 항목 | 주기 | 설명 |
|------|------|------|
| **봇 사이클** | 1-3초 | `MarketMaker.cycle()` 실행 |
| **ML 예측** | 매 사이클 | LightGBM 변동성/방향 예측 |
| **Adaptive VPIN** | 매 거래 | 동적 임계값 조정 |
| **Kill Switch** | 1시간 | 모듈별 PnL 체크 |

---

## 2. 모델 재학습 주기

| 모델 | 권장 주기 | 스크립트 |
|------|----------|----------|
| LightGBM 변동성 | 월 1회 | `python ml/train_lightgbm.py --target volatility` |
| LightGBM 방향 | 월 1회 | `python ml/train_lightgbm.py --target direction` |
| HMM 레짐 | 분기 1회 | `python ml/train_hmm.py` |
| RL Agent | 필요시 | `python rl/train_extended.py` (GPU 권장) |

---

## 3. 파라미터 튜닝 방법

### A. Paper Trading 기반 (현재)
```
Week 1: threshold 0.65 → 관찰
Week 2: 안정 시 → 0.75
Week 3: 안정 시 → 0.80
```

### B. 백테스트 기반 (권장) ⭐

**장점:**
- 빠른 검증 (1년 = 5분)
- 다양한 파라미터 조합 테스트
- 통계적 유의성 확보

**백테스트 튜닝 스크립트:**
```bash
# VPIN threshold 스캔
python backtest/run_microstructure_backtest.py --threshold 0.5,0.6,0.7,0.8

# 전체 파라미터 그리드 서치
python backtest/parameter_scan.py --config config_grid.yaml
```

**파라미터 스캔 예시:**
```yaml
# config_grid.yaml
scan:
  vpin_threshold: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
  defensive_risk_score: [0.8, 1.0, 1.2, 1.5]
  cautious_risk_score: [0.4, 0.6, 0.8]
```

---

## 4. 튜닝 일정 (권장)

| 주기 | 작업 | 방법 |
|------|------|------|
| **주간** | KillSwitch 상태 확인 | 대시보드 |
| **월간** | 모델 재학습 | 스크립트 실행 |
| **분기** | 전체 파라미터 백테스트 | 그리드 서치 |
| **반기** | PRD 및 전략 검토 | 수동 분석 |

---

## 5. 백테스트 vs Paper Trading

| 항목 | 백테스트 | Paper Trading |
|------|----------|---------------|
| 속도 | 빠름 (5분/년) | 느림 (실시간) |
| 현실성 | 제한적 | 높음 |
| 용도 | 파라미터 탐색 | 최종 검증 |
| 권장 | 초기 튜닝 | 라이브 전환 전 |

**권장 워크플로우:**
```
1. 백테스트로 최적 파라미터 후보 5개 선정
2. 상위 2개를 Paper Trading (각 7일)
3. 최종 선택 후 Live 전환
```

---

## 6. 자동화 상태

| 기능 | 상태 | 비고 |
|------|------|------|
| Kill Switch | ✅ 자동 | 모듈 비활성화 |
| Adaptive VPIN | ✅ 자동 | 임계값 조정 |
| Circuit Breaker | ✅ 자동 | 긴급 정지 |
| 모델 재학습 | ❌ 수동 | 스크립트 실행 필요 |
| 파라미터 튜닝 | ❌ 수동 | 백테스트 실행 필요 |

---

## 7. 드리프트 감지 (향후)

현재 `ml/drift_detector.py` 존재하나 알림 미구현.

**향후 추가 예정:**
- 주간 성과 리포트 자동 생성
- 모델 성능 저하 시 Telegram 알림
- 월간 재학습 스케줄러

---

## 관련 파일

- `ml/train_lightgbm.py` - LightGBM 학습
- `ml/train_hmm.py` - HMM 학습
- `rl/train_extended.py` - RL 학습
- `backtest/run_*_backtest.py` - 백테스트 스크립트
- `tools/vpin_analyzer.py` - VPIN 분포 분석
