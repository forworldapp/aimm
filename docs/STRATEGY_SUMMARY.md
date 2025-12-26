# GRVT Market Maker Strategy Summary

## 1. Core Philosophy (핵심 철학)
**"안전 제일, 추세 추종, 존버와 손절의 조화"**
*   평소(횡보장)에는 **존버(Profit Only)** 전략으로 절대 손해를 보지 않습니다.
*   추세(Trend)가 발생하면 **리스크 관리(Stop Loss)** 모드로 전환하여 큰 손실을 방지합니다.
*   변동성에 따라 그물망 간격을 자동으로 조절하여 최적의 체결 빈도를 유지합니다.

## 2. Order Logic (주문 로직)

### A. Grid Structure (그물망 구조)
*   **Type**: Trailing Grid (현재가 추적형). 시장 가격(Mid Price)을 중심으로 위아래로 호가를 생성합니다.
*   **Dynamic Spread (변동성 대응)**:
    *   **ATR(변동성 지표)**를 기반으로 스프레드 간격을 자동 조절합니다.
    *   **Safety Limit**: 아무리 변동성이 작거나 커져도 **최소 $100 ~ 최대 $200** 간격을 무조건 지킵니다. (잦은 체결 방지 및 안전 거리 확보).
*   **Order Size**: 매 주문마다 **$100 (USD)** 어치의 수량을 계산하여 진입합니다.

### B. Inventory Skew (재고 관리)
*   **Skew Factor**: **0.1% (Weak)**.
*   **Logic**: 재고(포지션)가 쌓일수록, 매도 목표가를 아주 조금씩 낮춰서 체결 확률을 높입니다.
*   **Max Limit**: 재고가 약 **$2,000 (주문금액 x 20회)** 쌓였을 때 Skew 효과가 최대치로 작동합니다.

## 3. Entry & Exit Strategy (진입 및 청산)

### A. Entry Anchor (진입 안전장치)
*   **원칙**: "내 평단가(Entry Price)보다 불리한 가격에 추가 매수하지 않는다." (물타기만 허용, 고점 추격 매수 방지).
*   단, **Profit Protection**으로 인해 평단가 아래로는 매도 주문을 내지 않는 것이 기본입니다.

### B. Regime-Based Risk Management (상황별 대응)

| 🚦 Regime (상태) | 🛡️ Tolerance (손절 허용) | 📝 행동 요령 |
| :--- | :--- | :--- |
| **NEUTRAL (횡보)** | **0.0% (존버)** | 절대 손절 없음. 무조건 평단가 위에서만 익절함. |
| **SIGNAL (추세)** | **0.5% (손절 가능)** | 추세가 불리하면 평단가 대비 -0.5% 가격에도 던져서 탈출 가능. |

### C. Signal Latch (신호 고정 - 스마트 유지)
*   **문제 해결**: 시그널이 잠깐 떴다가 사라져서 손절 기회를 놓치는 것을 방지.
*   **Logic**:
    1.  한 번 **Buy/Sell Signal**이 뜨면 그 상태를 **Latch(기억)**합니다.
    2.  이후 지표가 Neutral로 식더라도, **Risky Zone**에 머무르는 한 **Signal Mode(손절 가능)**를 유지합니다.
*   **Reset Condition (해제 조건)**:
    *   **RSI 안착**: Buy Latch는 **RSI > 40**이 되어야 해제. (확실한 반등 확인 전엔 손절 기능 유지).
    *   **RSI 안착**: Sell Latch는 **RSI < 60**이 되어야 해제.
    *   **Position Clear**: 포지션을 다 팔고(0) 나면 즉시 해제.

## 4. Summary of Current Parameters (현재 설정값)
*   `spread_pct`: **0.25%** (Base)
*   `process`: **Dynamic ($100 ~ $200)**
*   `order_size`: **$100**
*   `inventory_skew`: **0.1%**
*   `loss_tolerance`: **0.0% (Neutral) / 0.5% (Signal)**
*   `latch_rsi`: **40 / 60**
