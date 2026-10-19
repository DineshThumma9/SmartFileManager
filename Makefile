.PHONY: test

test:
	uv run --extra dev --offline --locked pytest -q
