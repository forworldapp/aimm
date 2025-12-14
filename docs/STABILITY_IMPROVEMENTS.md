# Stability & Process Management Improvements
## Summary of Issues and Resolutions (2025-12-14)

This document details the critical stability issues encountered during the bot's validation phase and the engineered solutions implemented to resolve them.

### 1. The "Zombie Process" Issue
**Symptoms:**  
- Dashboard displayed inconsistent data (e.g., Trade History shows "Buy", but Position is "0").
- UI flickering where metrics would disappear and reappear.
- Logs showed contradictory states.

**Root Cause:**  
- Multiple instances of `main.py` were running simultaneously in the background (Zombie Processes).
- These instances were competing to write to `paper_status.json` and `trade_history.csv`, causing Race Conditions and data corruption.

**Solution: Singleton Pattern with Socket Lock**  
- **Implementation**: Added a socket binding mechanism in `main.py`.
- **Mechanism**: Upon startup, the bot attempts to bind to `127.0.0.1:45432`.
- **Outcome**: If the port is already in use, the new instance immediately terminates with a "Bot is already running" error. This guarantees meaningful **Single Instance Enforcement**.

### 2. Dashboard Flickering (Race Conditions)
**Symptoms:**  
- Metrics like "Total Equity" or "Position" would occasionally flash to `0` or `N/A`.

**Root Cause:**  
- The bot was writing to `paper_status.json` at the exact same moment the dashboard was trying to read it. The dashboard would read an empty or partially written file.

**Solution A: Atomic Writes (Bot Side)**  
- **Ref**: `core/paper_exchange.py` (`_save_status`)
- **Mechanism**: The bot now writes data to a temporary file (`.tmp`) first, flushes it to disk, and then uses `os.replace()` to atomically swap it with the target file.
- **Benefit**: The target file is never in a "partial" or "empty" state during updates.

**Solution B: Session Persistence (Dashboard Side)**
- **Ref**: `dashboard.py`
- **Mechanism**: The dashboard now caches the last valid data packet in `st.session_state`.
- **Benefit**: If a file read fails for any reason, the dashboard gracefully falls back to the last known good state instead of showing zeros.

### 3. Process Control & Shutdown
**Symptoms:**  
- The "Stop" button only paused the strategy loop but left the Python process running, contributing to the zombie accumulation risk if restart commands were issued repeatedly.

**Solution: Shutdown Command**  
- **Feature**: Added a "💀 Shutdown Process" button to the Dashboard.
- **Mechanism**: Sends a `shutdown` command via `command.json`.
- **Logic**: The bot executes `sys.exit()` upon receiving this command, ensuring a clean and complete termination of the process.

---

## Validated Workflow
1. **Start**: Run `python main.py` (or use the Dashboard Start button if the process is already alive).
2. **Monitor**: Check Dashboard. Synchronization is now guaranteed.
3. **Stop**: Use "Stop & Close" to pause trading and exit positions.
4. **Terminate**: Use "Shutdown Process" to fully kill the application.
