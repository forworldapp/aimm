# CHANGELOG v5.1.0 — Bot-Only Risk Management

**Date**: 2026-02-15
**Version**: v5.1.0
**Previous Version**: v5.0.1 (commit `c3b7b88`)
**Current HEAD**: `ffdd4b1`

> ⚠️ **롤백 방법**: `git revert ffdd4b1..c3b7b88` 또는 `git reset --hard c3b7b88`

---

## 목적

리스크 관리(Max Drawdown, Circuit Breaker, Max Position, Inventory Skew 등)를
**전체 계정 포지션 기준 → 봇 전용 P&L 기준**으로 전환.
수동매매가 봇의 리스크 관리에 영향을 주지 않도록 분리.

---

## 커밋 내역 (6개, 시간순)

| # | Commit | 시간 | 제목 |
|---|--------|------|------|
| 1 | `17008a3` | 19:55 | fix(critical): add close_position to GrvtExchange |
| 2 | `8c9ea30` | 20:08 | feat: bot-only risk management with isolated P&L tracking |
| 3 | `84f5cfc` | 20:27 | feat: convert ALL risk management to bot-only P&L |
| 4 | `bd97cd8` | 20:49 | feat: persist bot order IDs to survive restarts |
| 5 | `3e91993` | 21:26 | fix(critical): remove broken order_id filter for trade detection |
| 6 | `ffdd4b1` | 21:55 | fix: prevent trade history duplication on restart |

---

## 변경 상세

### Commit 1: `17008a3` — close_position 추가
**파일**: `core/grvt_exchange.py`
**문제**: Max Drawdown 트리거 시 `close_position()` 메서드가 없어서 무한 에러 루프
**수정**:
- `close_position(symbol)` 메서드 추가
- Market order로 포지션 청산, 실패 시 공격적 limit order fallback
- 현재 포지션의 반대 방향으로 주문 발행

```python
# 추가된 메서드 (grvt_exchange.py)
async def close_position(self, symbol: str):
    # 1. 현재 포지션 크기/방향 확인
    # 2. 반대 방향 market order 발행
    # 3. 실패 시 orderbook 기반 limit order fallback
```

---

### Commit 2: `8c9ea30` — 봇 전용 P&L 추적 시작
**파일**: `core/grvt_exchange.py`, `strategies/market_maker.py`, `config.yaml`

#### grvt_exchange.py — `get_bot_pnl()` 추가
```python
def get_bot_pnl(self, symbol: str, current_price: float) -> dict:
    # trade_history CSV에서 봇 전용 P&L 계산
    # 반환: bot_net_qty, bot_avg_entry, realized_pnl, unrealized_pnl,
    #        total_pnl, trade_count, bot_cost_basis
```
- `trade_history_*.csv`에서 봇 매매 이력 읽기
- 실현 P&L = `grid_profit` 합계
- 미실현 P&L = (현재가 - 봇 평균진입가) × 봇 순포지션
- FIFO 방식으로 cost_basis 계산

#### market_maker.py — Circuit Breaker & Drawdown 변경
**이전**:
- Circuit Breaker: 계정 unrealizedPnL 기준, max_loss_usd
- Max Drawdown: 전체 equity의 5%

**이후**:
- Circuit Breaker: `bot_total_pnl < -$300` (봇 전용)
- Max Drawdown: `bot_total_pnl / bot_cost_basis > 15%` (봇 전용)

#### config.yaml
```yaml
# 변경
max_drawdown_pct: 0.05  →  0.15  (15% of bot cost basis)
```

---

### Commit 3: `84f5cfc` — 나머지 리스크 관리 전부 봇 전용 전환
**파일**: `strategies/market_maker.py`

10개 항목 변경 (`current_pos_qty` → `bot_pos_qty`):

| # | 항목 | 이전 (전체 계정) | 이후 (봇 전용) |
|---|------|-----------------|---------------|
| 1 | `self.inventory` | `position.amount` | `bot_net_qty` |
| 2 | Latch 리셋 조건 | `current_pos_qty == 0` | `bot_pos_qty == 0` |
| 3 | 상태 로그 | `Pos: / Equity:` | `BotPos: / BotP&L: (R:$ U:$)` |
| 4 | `inventory_ratio` (Skew) | 전체 포지션 비율 | 봇 포지션 비율 |
| 5 | Entry Anchor | `position.entryPrice` | `bot_avg_entry` |
| 6 | Max Position 한도 | `abs(current_pos_qty)*price` | `abs(bot_pos_qty)*price` |
| 7 | Funding Rate Integrator | 전체 inventory | 봇 inventory |
| 8 | Dynamic Order Sizer | 전체 `pos*price` | 봇 `pos*price` |
| 9 | Circuit Breaker | 전체 unrealizedPnL | 봇 total P&L |
| 10 | Max Drawdown | 전체 equity 5% | 봇 cost basis 15% |

**주의**: `current_pos_qty`는 여전히 실제 주문 관리(cancel, place)에 사용됨.
리스크 판단만 `bot_pos_qty` 기준.

---

### Commit 4: `bd97cd8` — Bot Order ID 영구 저장
**파일**: `core/grvt_exchange.py`
**문제**: `_bot_order_ids`가 메모리에만 저장 → 재시작 시 유실
**수정**:
- `data/bot_order_ids.json` 파일로 영구 저장
- `_load_bot_order_ids()`: 시작 시 파일에서 로드
- `_save_bot_order_ids()`: 주문 발행 시 즉시 저장
- `connect()`에서 startup reconciliation 로그

---

### Commit 5: `3e91993` — Order ID 필터 제거 (Critical)
**파일**: `core/grvt_exchange.py`
**문제**: GRVT SDK의 `create_order()`가 모든 주문에 `0x00` 반환
→ 체결 API의 실제 order_id와 불일치 → 모든 체결이 "수동매매"로 분류
**수정**:
- `fetch_and_save_trades()`에서 order_id 기반 필터 제거
- 이 서브어카운트의 모든 체결을 봇 체결로 기록
- 수동매매를 하지 않으므로 안전

```python
# 제거됨
if order_id not in self._bot_order_ids:
    skipped_manual += 1
    continue
```

---

### Commit 6: `ffdd4b1` — 체결 이력 중복 방지
**파일**: `core/grvt_exchange.py`
**문제**: CSV 리셋 후 `fetch_and_save_trades()`가 과거 50개 체결을 재삽입
→ 이전 세션 손실 합산 → Max Drawdown 오작동
**수정**:
- `data/processed_trade_ids.json` 파일 추가
- CSV와 독립적으로 처리 완료된 trade_id 기억
- CSV 리셋해도 과거 체결 재삽입 방지

---

## 신규/변경 파일 요약

| 파일 | 변경 내용 |
|------|----------|
| `config.yaml` | `max_drawdown_pct`: 0.05 → 0.15 |
| `core/grvt_exchange.py` | `close_position()`, `get_bot_pnl()`, Order ID 영구저장, 체결 중복 방지 |
| `strategies/market_maker.py` | 모든 리스크 관리를 `bot_pos_qty` 기준으로 전환 |

## 신규 데이터 파일

| 파일 | 용도 |
|------|------|
| `data/bot_order_ids.json` | 봇이 발행한 주문 ID 영구 저장 |
| `data/processed_trade_ids.json` | 처리 완료된 체결 ID (중복 방지) |

---

## 롤백 방법

### 전체 롤백 (v5.0.1로 복원)
```bash
git reset --hard c3b7b88
git push -f origin main
# data/ 폴더의 JSON 파일도 삭제 필요:
del data\bot_order_ids.json
del data\processed_trade_ids.json
# config.yaml의 max_drawdown_pct를 0.05로 수동 복원
```

### 부분 롤백 (특정 커밋만 되돌리기)
```bash
git revert <commit_hash>  # 특정 커밋만 되돌림
```

---

## 알려진 제약사항

1. **GRVT SDK `0x00` 문제**: `create_order()`가 항상 `0x00` 반환 → order_id 매칭 불가
   - 현재 우회: 모든 체결을 봇 체결로 기록
   - 영향: 수동매매 시 봇 P&L에 포함됨 (수동매매 안 하면 문제 없음)

2. **CSV 리셋 시 주의**: CSV를 리셋하려면 `processed_trade_ids.json`에 과거 trade_id를 먼저 등록해야 함

3. **봇 재시작 시 RSI 리셋**: 캔들 데이터가 초기화되어 RSI=50에서 시작 → 수 분간 부정확한 RSI
