.PHONY: tests

tests:
	uv run --isolated --extra dev --offline --locked pytest -q
