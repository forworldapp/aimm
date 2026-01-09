# Hybrid Exit Order Logic Design

## 현재 동작 (v2.2)

| Regime | Buy 호가 | Sell 호가 |
|--------|---------|----------|
| neutral | 5 layers | 5 layers |
| buy_signal | 5 layers | ❌ 없음 |
| sell_signal | ❌ 없음 | 5 layers |
| waiting | 유지 | 유지 |

---

## 하이브리드 로직 제안

### 핵심 아이디어
시그널 방향으로 진입하면서도, 기존 포지션은 청산할 기회 제공

### 케이스별 정리

#### Case 1: buy_signal + 롱 포지션 (동일 방향)
```
상황: 매수 시그널, 이미 롱 보유 중
목표: 롱 익절 기회 제공

호가:
├── Buy: 5 layers (추가 매수)
└── Sell: 1개 (진입가 + 0.25% 또는 Mid + spread 중 높은 가격)

청산 가격 계산:
exit_price = max(entry_price × 1.0025, mid_price × (1 + spread))
```

#### Case 2: buy_signal + 숏 포지션 (반대 방향)
```
상황: 매수 시그널, 숏 보유 중 (역방향)
목표: 숏 손절/청산

호가:
├── Buy: 5 layers (신규 매수) + 1개 (숏 청산)
└── Sell: ❌ 없음

청산 가격 계산:
exit_price = min(mid_price × (1 - spread), entry_price × 1.001)
```

#### Case 3: sell_signal + 숏 포지션 (동일 방향)
```
상황: 매도 시그널, 이미 숏 보유 중
목표: 숏 익절 기회 제공

호가:
├── Buy: 1개 (진입가 - 0.25% 또는 Mid - spread 중 낮은 가격)
└── Sell: 5 layers (추가 매도)

청산 가격 계산:
exit_price = min(entry_price × 0.9975, mid_price × (1 - spread))
```

#### Case 4: sell_signal + 롱 포지션 (반대 방향)
```
상황: 매도 시그널, 롱 보유 중 (역방향)
목표: 롱 손절/청산

호가:
├── Buy: ❌ 없음
└── Sell: 5 layers (신규 매도) + 1개 (롱 청산)

청산 가격 계산:
exit_price = max(mid_price × (1 + spread), entry_price × 0.999)
```

---

## 전체 매트릭스

| Regime | Position | Grid Orders | Exit Order | Exit Type |
|--------|----------|-------------|------------|-----------|
| buy_signal | 롱 (+) | Buy 5 | Sell 1 | 익절 |
| buy_signal | 숏 (-) | Buy 5+1 | - | 손절 |
| buy_signal | 없음 | Buy 5 | - | - |
| sell_signal | 숏 (-) | Sell 5 | Buy 1 | 익절 |
| sell_signal | 롱 (+) | Sell 5+1 | - | 손절 |
| sell_signal | 없음 | Sell 5 | - | - |
| neutral | Any | Buy 5 + Sell 5 | - | - |

---

## 청산 호가 스프레드 옵션

### Option 1: 고정 스프레드
```python
익절: entry_price × (1 ± 0.25%)
손절: mid_price × (1 ± spread)
```

### Option 2: ATR 기반 (변동성 반영)
```python
exit_spread = ATR × 0.5
익절: entry_price ± (ATR × 0.5)
```

### Option 3: 포지션 크기 기반
```python
대형 포지션 → 좁은 스프레드 (빠른 청산)
소형 포지션 → 넓은 스프레드 (수익 극대화)
```

---

## Neutral 복귀 시 동작

```
Signal → Neutral 전환 시:
1. 모든 청산 호가 취소
2. 양방향 그리드로 복원 (Buy 5 + Sell 5)
3. Smart Order Management로 불필요한 재배치 방지
```

---

## 구현 시 고려사항

1. **청산 수량**: 전체 포지션 vs 일부
2. **복수 청산 레이어**: 1개 vs 여러 가격대
3. **손절 트리거**: 시그널 기반 vs 손실 %
4. **Rate Limit**: 호가 변경 빈도 제한

---

## 테스트 시나리오

1. buy_signal 중 롱 포지션 청산 확인
2. sell_signal 중 롱 포지션 손절 확인  
3. Neutral 복귀 시 양방향 복원 확인
4. FIFO Grid Profit 계산 정확성
