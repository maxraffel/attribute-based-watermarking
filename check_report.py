"""Colored pass/fail reporting for script-style checks using Rich (console markup)."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel


def expect_cprf_ceval_ok(f: list[int], x: list[int], modulus: int) -> bool:
    """True iff constrained CPRF c_eval should agree with master eval (f·x ≡ 0 mod modulus)."""
    return sum(f[i] * x[i] for i in range(len(x))) % modulus == 0


class CheckReporter:
    """Accumulates boolean/metric checks and prints a Rich-styled summary."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(highlight=False)
        self._checks: list[bool] = []

    def section(self, title: str) -> None:
        self.console.print()
        self.console.rule(title, style="dim")

    def add_boolean(self, name: str, got: bool, expected: bool) -> None:
        ok = got == expected
        self._checks.append(ok)
        tag = "[bold green]PASS[/]" if ok else "[bold red]FAIL[/]"
        self.console.print(
            f"  {tag}  {name}  [dim](got {got}, expected {expected})[/]"
        )

    def add_metric(
        self,
        name: str,
        value: float,
        expected_high: bool,
        *,
        threshold: float = 0.99,
    ) -> None:
        high = value >= threshold
        ok = high == expected_high
        self._checks.append(ok)
        tag = "[bold green]PASS[/]" if ok else "[bold red]FAIL[/]"
        want = f"high (≥{threshold:.0%})" if expected_high else f"low (<{threshold:.0%})"
        self.console.print(
            f"  {tag}  {name}: [cyan]{value:.2%}[/]  [dim](expected {want})[/]"
        )

    def summary(self) -> tuple[int, int]:
        total = len(self._checks)
        passed = sum(self._checks)
        if total == 0:
            return 0, 0
        if passed == total:
            self.console.print(
                Panel(
                    f"[bold green]All checks passed ({passed}/{total})[/]",
                    border_style="green",
                    expand=False,
                )
            )
        else:
            self.console.print(
                Panel(
                    f"[bold red]Some checks failed ({passed}/{total} passed)[/]",
                    border_style="red",
                    expand=False,
                )
            )
        return passed, total
