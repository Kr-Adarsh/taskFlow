"""
Deterministic fault injection for Operon demo scenarios.
Enables controlled, visible transient failures (e.g. 503 before transaction commit)
that consume exactly once, allowing the agent to observe failure and recover.
"""

from typing import Optional

class FaultInjectionError(Exception):
    def __init__(self, message: str, error_code: str = "FINANCE_SERVICE_UNAVAILABLE", retriable: bool = True):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retriable = retriable

class FaultInjectionState:
    def __init__(self):
        self.active_targets: dict[str, int] = {}
        self.history: list[dict] = []

    def arm(self, target: str, count: int = 1) -> None:
        """Arms fault injection for a target (e.g., 'finance_create_invoice')."""
        self.active_targets[target] = count

    def maybe_fail(self, target: str) -> None:
        """
        If target is armed, decrements count and raises FaultInjectionError BEFORE transaction commit.
        """
        remaining = self.active_targets.get(target, 0)
        if remaining > 0:
            self.active_targets[target] = remaining - 1
            if self.active_targets[target] <= 0:
                del self.active_targets[target]
            
            error_record = {
                "target": target,
                "error_code": "FINANCE_SERVICE_UNAVAILABLE",
                "message": "Finance Service Unavailable: Transient lock acquisition timeout in ledger gateway. Transaction aborted.",
                "retriable": True
            }
            self.history.append(error_record)
            raise FaultInjectionError(
                message=error_record["message"],
                error_code=error_record["error_code"],
                retriable=True
            )

    def reset(self) -> None:
        """Resets all armed faults and history."""
        self.active_targets.clear()
        self.history.clear()

# Global singleton fault manager
fault_manager = FaultInjectionState()
