# Contributing

```bash
python -m pip install -e ".[dev]"
python -m ruff check .          # lint
python -m mypy src              # types
python -m pytest -q             # tests (network is blocked; API keys are blanked by tests/conftest.py)
python scripts/benchmark_pipeline.py   # reproducible pipeline timing with simulated latency
```

Rules of the road:

1. **Every fix ships with a regression test** that fails without the fix. Do not skip or loosen a test
   to get a green run; if a test encodes wrong behaviour, change it and say why in the commit.
2. **No secrets or real personal data** in code, fixtures, logs or `data/*.json` (use `env:NAME`).
3. **Anything sent to an LLM** goes through `fence()` / `fence_text()` (redacts PII, marks the text untrusted).
4. **Workflows** are linted with actionlint in CI; pipeline steps must fail the job when the work fails.
5. Performance work must be measured before and after with `scripts/benchmark_pipeline.py`.
