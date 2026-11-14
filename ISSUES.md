# Project issue tracker

Updated 2026-11-14 after a full project audit (security, reliability, product loop, docs). Earlier history is in git.

## Fixed in the 2026-11-14 audit

| Area | Issue | Fix |
| --- | --- | --- |
| Posting safety | Posts were auto-queued to Instagram with a queue limit of 8 and no per-day cap; manual re-runs stacked 10 posts in two days | Queue limit default is now 3 and at most one auto-queued post per 20 hours (`BUFFER_MIN_HOURS`); drafts, `--now`, `--prompt-id` and replacements are exempt |
| Site privacy | Docs implied `SITE_SECRET` made the hub private, but the repo and `gh-pages` branch are public and `data/site_manifests/` (with the keyed path) is on `main` | README, SECURITY.md, `.env.example` and the hub index page now say plainly: public while the repo is public, obscurity only |
| Site footer | Footer claimed "nothing is auto-posted" | Footer now says posts are queued to Buffer automatically when checks pass |
| LLM budget | Every Groq call reserved 8192 completion tokens, so prompts exceeded the 8000 TPM limit (HTTP 413) and variants failed | `completion_budget()` sizes `max_tokens` per call (label cap, provider TPM budget, model output cap) and skips models that cannot fit; settings `LLM_TPM_LIMITS`, `LLM_MODEL_OUTPUT_CAPS` |
| Silent degradation | Runs with unscored prompts finished green with no signal | Daily run reports stage `scoring` in the degraded-run issue and logs a warning |
| Stale alerts | Degraded-run and "workflow failed" issues stayed open for months | A clean daily run closes degraded-run issues; every alerting workflow closes its own failure alert after a green run |
| Lead pipeline | Zero yield was invisible; real-estate portals polluted "home decor" results | Rejection reasons are logged (and warned when everything is rejected); portal domains added to the ignore list |
| Learning path | `buffer_metrics.py` had 13% coverage | 17 tests cover scoring, snapshots, idempotent learning, maturity and freshness rules |
| Dependencies | `requirements.txt` was unpinned and omitted `huggingface_hub`; installs were not reproducible | `requirements.txt` is now a pinned lock; all installs use `-c requirements.txt`; a test keeps it in step with `pyproject.toml`; `pip-audit` audits the lock |
| Secrets exposure | Job-level secrets were visible to `pip install` | Dependency installs run with a scrubbed environment (`env -i`) |
| Scheduling | Feedback ran hourly in the shared concurrency group, which can displace a queued Daily run | Feedback cron is every 3 hours (issue events still trigger it immediately) |
| State merges | `-X theirs` could drop concurrent appends to `.jsonl` state | `.gitattributes` union-merges all three `.jsonl` state files |
| Dead config / hacks | Unused `*_CRON`, `PROMPTS_DIR` settings, wrong comments, `sys.path` hacks in module entry points | Removed or corrected |
| Accessibility | No focus styles, no skip link, filter chips without state | Focus-visible outlines, skip link, `aria-pressed` on chips, reduced-motion support |
| Docs drift | README said "daily 23:59 IST"; ISSUES.md described a removed feature set | README schedule corrected; this file rewritten |

## Configuration to set (repository settings, not code)

| Setting | Why |
| --- | --- |
| Secret `FEEDBACK_SIGNING_KEY` and variable `FEEDBACK_ALLOWED_USERS` | Without both, feedback links are inert and the feedback loop does nothing (fails closed) |
| Variable `BUFFER_DRAFT_MODE=true` (optional) | Switches the daily run from auto-queue to Buffer drafts for manual approval |

## Known gaps (deliberately not changed)

| Gap | Why it was left |
| --- | --- |
| Hub content is public while the repo is public | Product decision: keep the repo public. Make it private (and host the hub elsewhere) if content must not be public |
| State (`data/`) is committed to `main` by bots; no dedicated state branch | A branch migration needs a live cut-over that cannot be tested locally and risks losing learning state. Mitigated by union merges and the shared concurrency group |
| Long orchestration functions (`daily_job.run_for_business`, `virality_gate.run`, `lead_job.run`, `process_feedback`) and ~86 broad `except Exception` handlers | Behaviour-preserving splits need dedicated characterisation tests first; tracked as refactoring debt |
| No pagination on the hub page (118 entries today) | Retention caps growth at 45 days; revisit if the manifest passes ~1 MB |
| Leads pipeline yield | Diagnostics added; whether Apify/Instagram returns follower data for the niche cannot be verified offline |
| Learning needs 5 earlier mature comparable posts per business before the first Buffer-metrics update | By design (avoids learning from noise); with one post per day, expect several weeks |
| Live hosting of the hub (Pages API returns 404) | Cannot be verified from the repo; confirm where `gh-pages` is served |

## Verification limits

Workflows were validated with `actionlint` and the test suite, not by running them on GitHub. The first scheduled Daily, Leads and feedback runs after merge are the real test for: the pinned install (`-c requirements.txt`), the scrubbed-environment `pip install`, the token budget on Groq, and the auto-close steps.
