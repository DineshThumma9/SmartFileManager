.PHONY: test tests

test:
	uv run --extra dev --offline --locked pytest -q

tests: test
