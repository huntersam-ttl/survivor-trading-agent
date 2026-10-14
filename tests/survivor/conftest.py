import pytest


@pytest.fixture(autouse=True)
def explicit_paper_opt_in_for_hermetic_survivor_tests(monkeypatch):
    """Paper-cycle tests run only against isolated fakes and temp ledgers."""
    monkeypatch.setenv("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", "true")
