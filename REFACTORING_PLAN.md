# GRVT 마켓메이킹 봇 - 리팩터링 플랜
**분석일**: 2026-02-23 | **기준 버전**: v7.0.2

> [!CAUTION]
> 서킷 브레이커 결함과 주문 검증 부재는 실제 자금 손실로 이어질 수 있으므로 즉각 수정이 필요합니다.

---

## 🎯 우선순위 요약

| 우선순위 | 항목 | 파일 | 핵심 변경 |
|---|---|---|---|
| 🔥 즉시 | 서킷 브레이커 포지션 청산 | `market_maker.py:944` | `close_position()` 추가 |
| 🔥 즉시 | 주문 검증 | `grvt_exchange.py:112` | `_validate_order()` 메서드 추가 |
| 🔥 즉시 | bare `except:` 제거 | `market_maker.py:915`, `paper_exchange.py:166` | 구체적 예외 타입 지정 |
| ⚡ 높음 | unreachable 코드 | `market_maker.py:374` | 죽은 코드 2줄 삭제 |
| ⚡ 높음 | 설정 검증 | `core/config.py` | `_SCHEMA + validate()` 추가 |
| ⚡ 높음 | 원자적 파일 쓰기 | `paper_exchange.py` | `tempfile + os.rename()` 패턴 |
| 📊 중간 | `requirements.txt` | `requirements.txt` | 누락 패키지 + 버전 핀 추가 |
| 📊 중간 | 테스트 | `tests/` | 서킷브레이커/검증 테스트 추가 |
| 📖 낮음 | God Class 분리 | `market_maker.py` | `OrderManager`, `MLManager`, `StatusPublisher` |

---

## 🔴 치명적 문제 (즉시 수정)

### 1. 서킷 브레이커 결함 — `market_maker.py:944`

**문제**: 손실 한도 초과 시 주문만 취소하고 **포지션은 유지**. 봇 정지 후에도 미실현 손실 계속 발생.

```python
# ✅ 수정 후: cancel_all_orders() 뒤에 포지션 청산 추가
await self.exchange.cancel_all_orders(self.symbol)

# ★ 추가 (기존에 없던 부분)
if abs(bot_pos_qty) > 0:
    self.logger.critical(f"   Closing position: {bot_pos_qty:.4f} @ market price")
    try:
        await self.exchange.close_position(self.symbol)
        self.logger.critical("   Position close order submitted.")
    except Exception as close_err:
        self.logger.critical(f"   FAILED to close position: {close_err}")
```

> [!IMPORTANT]
> MaxDrawdown 트리거(`market_maker.py:963`)에도 동일하게 적용해야 합니다.

---

### 2. 주문 검증 부재 — `grvt_exchange.py:112`

**문제**: 최소 수량, 틱 사이즈, 마진 검증 없이 주문 제출 → 거래소 거부 또는 비정상 체결.

```python
# ✅ GrvtExchange 클래스에 추가할 상수 및 메서드
MIN_ORDER_QTY = {'BTC_USDT_Perp': 0.001, 'ETH_USDT_Perp': 0.01}
TICK_SIZE     = {'BTC_USDT_Perp': 0.1,   'ETH_USDT_Perp': 0.01}

def _validate_order(self, symbol: str, price: float, quantity: float) -> tuple[bool, str]:
    min_qty = self.MIN_ORDER_QTY.get(symbol, 0.001)
    tick    = self.TICK_SIZE.get(symbol, 0.1)

    if quantity < min_qty:
        return False, f"Quantity {quantity} below minimum {min_qty}"
    if price <= 0:
        return False, f"Price must be positive, got {price}"

    # 틱 사이즈 정렬 확인
    rounded = round(round(price / tick) * tick, 10)
    if abs(rounded - price) > tick * 0.01:
        return False, f"Price {price} not aligned to tick size {tick}"

    return True, ""

# ✅ place_limit_order() 앞에 검증 삽입
async def place_limit_order(self, symbol, side, price, quantity):
    is_valid, err = self._validate_order(symbol, price, quantity)
    if not is_valid:
        self.logger.error(f"Order validation failed: {err}")
        return None
    # ... 기존 주문 로직 ...
```

---

### 3. bare `except:` 제거

**문제**: `except:` 는 `KeyboardInterrupt`, `SystemExit`까지 삼킴 → 봇 강제 종료 불가.

```python
# ❌ 변경 전 (market_maker.py:915, paper_exchange.py:166)
except:
    return

# ✅ 변경 후
except (KeyError, IndexError, TypeError, ValueError) as e:
    self.logger.warning(f"Failed to parse orderbook: {e}")
    return
```

---

## 🏗️ 아키텍처 문제

### God Class — `market_maker.py` (1,827줄)

15개 이상의 책임이 단일 클래스에 집중. 장기 목표로 분리 권장:

```
MarketMaker (조율자)
├── OrderManager        — 주문 생성/취소/관리
├── MLManager           — 레짐 감지, 파라미터 블렌딩
├── RiskManager         — 포지션 한도, 서킷 브레이커
└── StatusPublisher     — 대시보드 JSON 저장, 텔레그램 알림
```

### LSP 위반 — `paper_exchange.py:30`

```python
# ❌ 현재: 부모 생성자 미호출
class PaperGrvtExchange(GrvtExchange):
    def __init__(self):
        # Do NOT call super().__init__()  ← Liskov 치환 원칙 위반
```

장기적으로는 `ExchangeInterface` 추상 베이스 클래스를 두고 `GrvtExchange`와 `PaperGrvtExchange`를 별도 구현체로 리팩터링 권장.

### 중복 코드

- `market_maker.py:1536-1572` vs `1272-1295` — 메트릭 푸시 로직 중복
- `grvt_exchange.py` vs `paper_exchange.py` — CSV 작성 로직 중복

---

## ⚡ 높음 (1–2주)

### 4. unreachable 코드 — `market_maker.py:374`

```python
# ❌ 변경 전 (실행 불가 코드 존재)
return BollingerFilter(...)
conf = Config.get("strategy", "rsi", {})  # UNREACHABLE
return RSIFilter(...)                      # UNREACHABLE

# ✅ 변경 후: 순서 정리, 불필요한 줄 2개 삭제
if name == 'bollinger':
    conf = Config.get("strategy", "bollinger", {})
    return BollingerFilter(conf.get('period', 20), conf.get('std_dev', 2.0))
if name == 'rsi':
    conf = Config.get("strategy", "rsi", {})
    return RSIFilter(conf.get('period', 14), ...)
return None
```

---

### 5. 설정 검증 — `core/config.py`

```python
# ✅ 스키마 기반 검증 추가
_SCHEMA = {
    "strategy": {
        "spread_pct":     (float, 0.0001, 0.1,   True),
        "grid_layers":    (int,   1,      20,     True),
        "order_size_usd": (float, 1.0,    100000, True),
    },
    "risk": {
        "max_position_usd": (float, 1.0, 1e9, True),
        "max_loss_usd":     (float, 0.0, 1e9, True),
    }
}

@classmethod
def validate(cls) -> list[str]:
    errors = []
    for section, fields in _SCHEMA.items():
        for key, (typ, min_v, max_v, required) in fields.items():
            value = cls._config.get(section, {}).get(key)
            if value is None and required:
                errors.append(f"[{section}.{key}] is required but missing")
                continue
            if value is not None and not isinstance(value, typ):
                errors.append(f"[{section}.{key}] expected {typ.__name__}, got {type(value).__name__}")
    return errors
```

---

### 6. 원자적 파일 쓰기 — `paper_exchange.py`

```python
# ✅ 크래시 시 파일 손상 방지
import tempfile

def _save_status(self):
    status = { ... }
    try:
        dir_name = os.path.dirname(self.status_file)
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_name,
                                         delete=False, suffix='.tmp') as tmp:
            json.dump(status, tmp)
            tmp_path = tmp.name

        if os.path.exists(self.status_file):
            os.remove(self.status_file)   # Windows 호환
        os.rename(tmp_path, self.status_file)
    except Exception as e:
        self.logger.warning(f"Failed to save status: {e}")
```

---

## 📊 중간 (1개월)

### 7. `requirements.txt` 의존성 고정

```
# 현재 누락 항목 추가 + 버전 핀
streamlit==1.35.0
streamlit-autorefresh==1.0.1
plotly==5.22.0
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
ccxt==4.3.32
python-dotenv==1.0.1
pyyaml==6.0.1
aiohttp==3.9.5
```

---

### 8. 테스트 추가

**현재 미테스트 핵심 항목:**
- `MarketMaker.cycle()` — 전략 핵심 루프
- 서킷 브레이커 동작
- PnL 계산 로직 (`FIFO 그리드 수익`)
- 페이퍼/라이브 모드 전환

```python
# tests/test_circuit_breaker.py (예시)
@pytest.mark.asyncio
async def test_circuit_breaker_closes_position(mock_exchange):
    """서킷 브레이커 발동 시 포지션 청산 확인."""
    mock_exchange.cancel_all_orders.assert_called_once()
    mock_exchange.close_position.assert_called_once_with("BTC_USDT_Perp")
```

---

## 💾 상태 관리 문제

재시작 후 복구 현황:

| 항목 | 현재 상태 |
|---|---|
| 잔고 & 포지션 | ✅ 복구됨 |
| 미체결 주문 | ❌ 손실 |
| FIFO 큐 | ❌ 초기화 |
| 누적 그리드 수익 | ❌ 초기화 |
| 상태 파일 원자성 | ❌ 크래시 시 손상 가능 |

> [!NOTE]
> 장기적으로 SQLite 사용을 권장하며, 단기적으로는 원자적 파일 쓰기(항목 6)로 가장 큰 위험 완화 가능.

---

## 📖 낮음 (지속)

- **CLAUDE.md 업데이트**: `v1.4.1` → `v7.0.2`로 버전 동기화
- **API 문서화**: 핵심 메서드에 docstring + 타입 힌트 추가
- **아키텍처 다이어그램**: 데이터 흐름도 작성
