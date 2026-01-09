# GRVT Bot 실행 가이드

## 폴더 구조

| 폴더 | 용도 | config.yaml mode |
|------|------|------------------|
| `grvt_bot` | **Live Trading** | `live` |
| `grvt_bot_paper` | Paper Trading | `paper` |

---

## 1. Live Trading 봇 실행

```powershell
cd c:\Users\camel\.gemini\antigravity\scratch\grvt_bot
python main.py
```

## 2. Paper Trading 봇 실행

```powershell
cd c:\Users\camel\.gemini\antigravity\scratch\grvt_bot_paper
python main.py
```

---

## 대시보드 실행

**Live (포트 8503):**
```powershell
cd c:\Users\camel\.gemini\antigravity\scratch\grvt_bot
python -m streamlit run dashboard.py --server.port 8503
```

**Paper (포트 8504):**
```powershell
cd c:\Users\camel\.gemini\antigravity\scratch\grvt_bot_paper
python -m streamlit run dashboard.py --server.port 8504
```

**접속:** http://localhost:8503 또는 http://localhost:8504

---

## 봇 중단

- **Ctrl + C** (해당 터미널에서)
- 또는 전체 중지:
```powershell
Get-Process python | Stop-Process -Force
```

---

## 데이터 초기화 (필요 시)

```powershell
Remove-Item data\*.csv -ErrorAction SilentlyContinue
Remove-Item data\*.json -ErrorAction SilentlyContinue
```

---

## 주의사항

- Live/Paper 동시 실행 시 **다른 포트** 사용
- 봇 실행 전 `.env` 파일에 API 키 설정 확인
- `waiting` → `neutral` 전환까지 약 15분 소요
