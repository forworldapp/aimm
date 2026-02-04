"""
Module Kill Switch - v5.2.1
Per-module PnL tracking and automatic disable when loss threshold breached
"""
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModulePerformance:
    """Track individual module performance"""
    name: str
    enabled: bool = True
    pnl: float = 0.0
    trades: int = 0
    last_update: float = field(default_factory=time.time)
    disabled_reason: Optional[str] = None
    disabled_at: Optional[float] = None


class KillSwitch:
    """
    Per-module kill switch to disable underperforming modules
    
    Unlike Circuit Breaker (全体 봇 停止), this only disables individual modules
    while allowing the rest of the system to continue operating.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("KillSwitch")
        
        # Global settings
        self.enabled = self.config.get('enabled', True)
        self.check_interval_seconds = self.config.get('check_interval_minutes', 60) * 60
        
        # Module tracking
        self.modules: Dict[str, ModulePerformance] = {}
        self.last_check = time.time()
        
        # Default limits per module type
        self.default_limits = {
            'microstructure_signals': 200,
            'cross_asset_hedge': 300,
            'order_flow_analysis': 150,
            'funding_rate_arbitrage': 250,
            'execution_algo': 100,
            'rl_agent': 200,
        }
        
    def register_module(self, name: str, max_loss_usd: float = None):
        """Register a module for tracking"""
        if name not in self.modules:
            self.modules[name] = ModulePerformance(name=name)
            limit = max_loss_usd or self.default_limits.get(name, 200)
            self.logger.info(f"📊 KillSwitch: Registered '{name}' (limit: ${limit})")
    
    def record_pnl(self, module_name: str, pnl_delta: float, trade_count: int = 1):
        """Record PnL contribution from a module"""
        if module_name not in self.modules:
            self.register_module(module_name)
        
        mod = self.modules[module_name]
        mod.pnl += pnl_delta
        mod.trades += trade_count
        mod.last_update = time.time()
        
        # Check if module should be disabled
        self._check_module(module_name)
    
    def _check_module(self, module_name: str):
        """Check if module should be disabled"""
        if not self.enabled:
            return
        
        mod = self.modules.get(module_name)
        if not mod or not mod.enabled:
            return
        
        limit = self.config.get(f'{module_name}_limit', self.default_limits.get(module_name, 200))
        
        if mod.pnl < -limit:
            self._disable_module(module_name, f"PnL loss exceeded ${limit} (current: ${mod.pnl:.2f})")
    
    def _disable_module(self, module_name: str, reason: str):
        """Disable a module"""
        mod = self.modules.get(module_name)
        if not mod:
            return
        
        mod.enabled = False
        mod.disabled_reason = reason
        mod.disabled_at = time.time()
        
        self.logger.warning(f"🛑 KillSwitch: Disabled '{module_name}' - {reason}")
        self.logger.warning(f"   Stats: PnL=${mod.pnl:.2f}, Trades={mod.trades}")
    
    def is_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled"""
        mod = self.modules.get(module_name)
        if not mod:
            return True  # Unknown modules are enabled by default
        return mod.enabled
    
    def get_status(self) -> Dict:
        """Get status of all modules"""
        return {
            name: {
                'enabled': mod.enabled,
                'pnl': round(mod.pnl, 2),
                'trades': mod.trades,
                'disabled_reason': mod.disabled_reason,
            }
            for name, mod in self.modules.items()
        }
    
    def reset_module(self, module_name: str):
        """Reset module PnL and re-enable"""
        mod = self.modules.get(module_name)
        if mod:
            mod.enabled = True
            mod.pnl = 0.0
            mod.trades = 0
            mod.disabled_reason = None
            mod.disabled_at = None
            self.logger.info(f"♻️ KillSwitch: Reset '{module_name}'")
    
    def periodic_check(self):
        """Run periodic check on all modules"""
        now = time.time()
        if now - self.last_check < self.check_interval_seconds:
            return
        
        self.last_check = now
        
        for name in self.modules:
            self._check_module(name)
        
        # Log summary
        enabled_count = sum(1 for m in self.modules.values() if m.enabled)
        total_pnl = sum(m.pnl for m in self.modules.values())
        self.logger.info(f"📊 KillSwitch Check: {enabled_count}/{len(self.modules)} modules active, Total PnL: ${total_pnl:.2f}")


# Global instance
_kill_switch: Optional[KillSwitch] = None


def get_kill_switch(config: dict = None) -> KillSwitch:
    """Get or create global KillSwitch instance"""
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch(config)
    return _kill_switch


def check_module_enabled(module_name: str) -> bool:
    """Quick check if module is enabled"""
    if _kill_switch is None:
        return True
    return _kill_switch.is_enabled(module_name)
