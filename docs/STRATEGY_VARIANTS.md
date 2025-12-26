# Strategy Variants: Grid Accumulator vs. Trend Sniper

## 📉 v1.4.2: Grid Accumulator (Legacy)
**"횡보장의 제왕, 추세장은 존버"**

*   **진입(Entry)**:
    *   **Neutral**: 적극적으로 양방향(Long/Short) 그리드 주문을 냄.
    *   **Signal**: 추세 방향으로 가속 진입.
*   **청산(Exit)**:
    *   **Neutral**: 이익 실현(Profit Only). 손절 없음(0%).
    *   **Signal**: 약손절(0.5%) 또는 추세 추종.
*   **장점**: 횡보장에서 시세 차익을 계속 누적하여 높은 회전율을 보임.
*   **단점**: 횡보하다가 강한 추세가 터지면, 반대 포지션에 물린 채로 시드(Seed)가 묶여버림. (존버 리스크).

---

## 🎯 v1.5.0: Trend Sniper (Current)
**"확실할 때만 쏜다"**

*   **진입(Entry)**:
    *   **Neutral**: **매매 금지 (No Trade)**. 신규 진입을 절대 하지 않음.
    *   **Signal**: 오직 확실한 추세 신호(Bollinger Breakout, Combo 등)가 떴을 때만 진입.
*   **청산(Exit)**:
    *   **Neutral**: 기존 포지션이 있다면 **청산(Reduce Only)** 주문만 허용. (Open 금지).
    *   **Signal**: 추세가 꺾이거나 RSI가 과열되면 이익 실현.
*   **장점**: 쓸데없이 물려있는 시간을 최소화하고, 승률 높은 구간만 공략. 현금 보유 비중이 높음.
*   **단점**: 지루한 횡보장에서는 거래가 전혀 없어 심심함.

## ⚙️ 설정 방법 (How to Switch)
현재 코드는 **v1.5.0 (Trend Sniper)** 로 설정되어 있습니다.
v1.4.2 방식으로 돌아가려면 `market_maker.py`의 Neutral Logic 부분을 수정해야 합니다.
