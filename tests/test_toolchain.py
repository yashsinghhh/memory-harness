"""Stage 0 gate: the suite runs, and async tests run without a decorator."""


def test_suite_runs() -> None:
    assert True


async def test_async_works_without_a_decorator() -> None:
    """Proves asyncio_mode = "auto" is active. Without it this is skipped, not run."""
    assert True
