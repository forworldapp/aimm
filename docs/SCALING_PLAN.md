# GRVT Bot 점진적 스케일업 계획

## 현재 설정 (Phase 1)

| 파라미터 | 값 |
|----------|-----|
| `order_size_usd` | $100 |
| `max_position_usd` | $500 |
| `max_drawdown_pct` | 5% |

---

## Phase 2: 첫 확장 (2-3주차)

**조건**: 1주차 순이익 달성 + 시스템 안정

| 파라미터 | 변경 후 |
|----------|---------|
| `order_size_usd` | $150 |
| `max_position_usd` | $750 |
| `max_drawdown_pct` | 7% |

---

## Phase 3: 정상 운영 (1개월 후)

| 파라미터 | 변경 후 |
|----------|---------|
| `order_size_usd` | $200 |
| `max_position_usd` | $1,000 |
| `max_drawdown_pct` | 10% |

---

## Phase 4: 고급 운영 (3개월 후)

| 파라미터 | 변경 후 |
|----------|---------|
| `order_size_usd` | $300-500 |
| `max_position_usd` | $2,000-3,000 |
| `max_drawdown_pct` | 15% |

---

## 스케일업 규칙

| 조건 | 조치 |
|------|------|
| 일주일 순이익 > 2% | ➡️ 다음 Phase |
| 일주일 순손실 > 5% | ⬅️ 이전 Phase 롤백 |
| 시스템 에러 | ⏸️ 중지 후 분석 |

---

## 비상 대응

1. **봇 중지**: `Ctrl+C`
2. **포지션 청산**: GRVT 웹에서 직접
3. **롤백**: `git checkout v1.9.4`
