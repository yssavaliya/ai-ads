# Self-Learning Business Content Intelligence Platform

Analyzes businesses (website + social handles), researches audience & competitors,
generates **viral-engineered content prompts** (scored 0-100, refined until ≥ threshold),
**renders them into real ad assets** — feed images with burned-in hooks and 9:16 Reel
MP4s with voiceover — and publishes them daily to a **static website (the Content Hub)** for a human to review and post
(**no auto-posting**). Runs 100% in GitHub Actions with free & open-source tooling only.
Can also **discover businesses automatically** and generate sample creatives for them.

## Lead discovery (no manual business entry)

The platform hunts for businesses that need content help:

1. **Discover**: searches web (DuckDuckGo/Bing, jittered + throttled) + Apify Instagram
   search for businesses in your configured niches (`data/lead_config.json`); repeat
   discoveries within 21 days are skipped (`data/lead_seen.json`)
2. **Enrich**: fetches each candidate's website, extracts Instagram/LinkedIn handles
   and contact emails (only real mailto: addresses — no placeholder emails)
3. **Score**: opportunity score 0-100 — active + real following + **low engagement rate**
   = they need viral content help (score high). Too small, too big (has agency), or
   already high-engagement → excluded
4. **Generate + render**: for the top N leads, runs the full pipeline (profile → audience
   → trends → virality gate) and renders finished sample creatives (image inline, Reel MP4 attached)
5. **Deliver**: saves `data/leads.json` + publishes the digest page (scores, reasons,
   rendered creatives, "no contact email found" flags where applicable)

Run weekly (Monday, `leads.yml`) or manually from the Actions tab. Configure niches,
follower range, target count, and digest email in `data/lead_config.json`.

## How it works

```
23:59 IST daily (18:29 UTC) ──▶ daily.yml
  0. AUTO-LEARN: scrape the business's own posts → match to previously sent
     prompts (exact post ID when the user tapped 📎 "I posted this", else
     date + caption similarity) → score real engagement → update Bayesian weights
  1. Scrape website profile (httpx)
  2. Fetch social data + competitor posts (Apify free tier / instaloader)
  3. Audience Mapper → personas, pain points, themes
  4. Trend Radar → today's topics, hashtags, gaps, viral patterns
  5. Virality Gate: generate → predict → if score < 75, refine with hints → repeat
  6. Creative Forge → render approved prompts into images/Reels (Pollinations +
     Pillow overlay; Reels add Piper TTS voiceover + FFmpeg motion assembly)
  7. Publish the finished assets to the Content Hub (GitHub Pages) with 👍/👎/🔄/📎 feedback buttons

Sunday 8 AM IST (02:30 UTC) ──▶ weekly.yml
  Top creatives + real engagement metrics + why · audience shifts ·
  competitor moves · next week calendar

Every 6h ──▶ engagement.yml  (scrape own posts → auto-learn)

Hourly ──▶ feedback.yml
  Parse feedback issues → Bayesian weight updates → close issues
```

## How auto-learning works (no clicks required)

The system learns automatically from real post performance:

1. **Daily**: the business's own Instagram posts are scraped (Apify free tier)
2. **Matching**: posts linked via the site's **📎 "I posted this"** button are matched
   by exact post ID (no guessing). Unlinked posts fall back to date window
   (+21 days) + caption/prompt similarity (≥ 0.30 Jaccard/containment)
3. **Scoring**: engagement rate = (likes + 2·comments + 3·shares + 2·saves) / reach
4. **Benchmark**: a post is "good" if its rate ≥ the business's own median,
   "bad" otherwise
5. **Update**: outcomes feed the Bayesian weight loop (`data/weights.json`) across
   prompt type × format × audience segment, so the next round favors what worked

Manual 👍/👎/🔄 clicks remain as additional signals. Run the auto-learner standalone
with `python scripts/run_auto_learn.py`.

## Requirements (GitHub Secrets)

| Secret | Purpose |
|--------|---------|
| `GROQ_API_KEY`, `GROQ_MODEL` | Primary LLM, free tier (console.groq.com) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Fallback LLM, free tier (aistudio.google.com) |
| `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_HOST`, `SMTP_PORT` | Only for the legacy e-mail mode (`DELIVERY_MODE=email\|both`) |
| `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY`, `MISTRAL_API_KEY` | Optional extra free LLM fallbacks |
| `TOGETHER_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` | Optional extra free image fallbacks |
| `SITE_SECRET` | Optional: makes site URLs unguessable and hides the index (see *Content Hub*) |
| `DELIVERY_MODE` (repo *variable*) | `site` (default), `email` or `both` |
| `APIFY_TOKEN` | Free tier social scraping (console.apify.com) |
| `HF_TOKEN` | Optional image fallback (Hugging Face FLUX.1-schnell) |
| `GITHUB_TOKEN` | Feedback issues + degraded-run tracking (built-in) |
| `FEEDBACK_ALLOWED_USERS` (repo *variable*, optional) | Comma-separated GitHub logins whose feedback issues count; unset = anyone who can open an issue |

LLM chain: Groq → Gemini (if both fail the run is reported as degraded — there is no keyless text fallback).
Images: Pollinations (keyless) → HF fallback. No local models run on CPU.
If any stage fails, a `[pipeline] degraded run — <date>` tracking issue is
created/updated so green CI never hides a dead run.

## Quick start

```bash
# Local dev
pip install -e .
python scripts/init_db.py
python scripts/onboard_business.py --name "Acme" --website "https://acme.com" --email "me@acme.com"

# Dry run locally: publishes the site to ./site (open site/index.html), sends nothing
DRY_RUN=true python -m src.scheduler.daily_job
DRY_RUN=true python -m src.scheduler.lead_job

# Trigger a run manually
gh workflow run daily.yml
gh workflow run leads.yml -f niches="bakery,home decor" -f target_count=5
```

Open `site/index.html` in a browser (or `python -m http.server -d site`) to inspect the actual site;
images and MP4s land in `logs/assets/`.

## Repo layout

```
.github/workflows/   daily · weekly · feedback · engagement · onboard · leads (shared concurrency group) · ci · drift_check
assets/fonts/        Poppins Bold (OFL license) for hook overlays
src/fetch/           website scraper (incl. mailto extraction), social scraper
                     (Apify → instaloader), competitor finder, lead finder
                     (jittered search + cooldown cache)
src/intelligence/    audience mapper, trend radar, virality predictor, lead scorer
src/generation/      prompt forge (with reel_beats), virality gate, creative forge
src/delivery/        delivery channels (static site publisher; legacy e-mail sender), github feedback links,
                     degraded-run tracker
src/learning/        bayesian updater, feedback processor (incl. POSTED links), auto-learner
src/scheduler/       daily_job, weekly_job, lead_job
src/storage/         SQLite models (agent.db lives in Actions cache, NOT git)
src/shared/          config, LLM client (Groq/Gemini), utils
templates/           daily_prompt.html, weekly_report.html, welcome.html, leads_digest.html
data/                businesses.json, lead_config.json, lead_seen.json, lead_rejected.json, weights.json (committed)
```

Rendered assets are published to the `gh-pages` branch (one orphan commit per publish — no history bloat) and uploaded as workflow artifacts. They are never committed to `main`.

## Onboarding a business

Use the `onboard.yml` workflow (Actions → Onboard Business → Run workflow) or run the
script locally. Demo seed records carry `"demo": true` and are skipped by scheduled runs.
Feedback buttons on the site open GitHub issues that the hourly feedback job turns into
Bayesian weight updates; the **📎 I posted this** button additionally links the published
post URL directly to the asset for exact engagement attribution. Posting alone is
not counted as a success: the learner scores that post's real engagement later.

## Content Hub (the static site)

E-mail is replaced by a small static website published to GitHub Pages
(`https://<owner>.github.io/<repo>/`, or the URLs printed in each run's summary).

- **What you get per business:** every prompt with its finished image or Reel (download
  buttons), the copy-paste caption, hashtags, best time to post, first comment, the full
  creative brief, why it should work (strengths/risks), the 👍/👎/🔄/📎 feedback buttons,
  a local "posted" tracker, search and type filters. Plus weekly deep-dives, an
  "about this setup" panel and a banner whenever a day's generation failed.
- **Leads:** the weekly digest has its own page.
- **How it is built:** each pipeline restores the `gh-pages` branch into `site/`, adds
  its entries (images are re-encoded to web JPEGs; Reels are copied if < 25 MB), prunes
  entries older than `SITE_RETENTION_DAYS` (45) and force-pushes a single orphan commit.
  The JSON manifests in `site/data/` are the source of truth; the HTML is regenerated.
- **Enable once:** *Settings → Pages → Build and deployment → Deploy from a branch →
  `gh-pages` / `/ (root)`*. Nothing else is needed; the first publish creates the branch.
- **Privacy:** GitHub Pages sites are public. Contact e-mail addresses are never
  published and pages are `noindex`. Set the secret `SITE_SECRET` and every page lives
  at an unguessable keyed URL while the index lists nothing (lead e-mails then show up
  too). The URLs appear in each run's summary. For truly private hosting use a private
  repo on a plan with private Pages.
- **E-mail still works:** `DELIVERY_MODE=email` or `both` (repo variable) re-enables the
  legacy SMTP e-mails.

## Free fallback chain (limits, outages, retired models)

Nothing hard-depends on one provider. Every call walks an ordered chain and moves on
when a provider is rate limited, out of quota, unauthorised, retired or returns unusable
output; failed providers/models are skipped for a while (circuit breaker) instead of
being retried, and if everything is merely rate limited the client waits once for the
shortest `Retry-After`.

| | Order (configurable) | Needs |
|---|---|---|
| **Text** (`LLM_PROVIDER_ORDER`) | Groq → Gemini → GitHub Models (opt-in) → Cerebras → OpenRouter (`:free` models) → Mistral → Hugging Face → keyless Pollinations | Each is optional: only providers with a key take part. GitHub Models (`GITHUB_MODELS_ENABLED=true`, built-in token, `models: read`) is opt-in: it answered with a plain `OK` from Actions when tried. Set `LLM_KEYLESS_FALLBACK=false` to keep prompts off the public keyless endpoint. |
| **Images** (`IMAGE_PROVIDER_ORDER`) | Pollinations → Gemini image → Hugging Face FLUX → Together FLUX → Cloudflare FLUX → **local branded card** | The card is rendered locally and cannot fail, so a post always ships with a visual (flagged with a warning). `IMAGE_CARD_FALLBACK=false` disables it. |
| **Voiceover** | Piper → silent Reel → still image | |

Model lists are plain env vars (`GROQ_FALLBACK_MODELS`, `OPENROUTER_MODELS`, …): free
model names rotate, and an unknown one is simply skipped. If a provider answers with
invalid JSON the next provider is asked (`run_json`). The daily *Drift Check* probes every
configured provider and opens an issue only if **all** of them fail.

## Keeping personal data out of git

Any string value in `data/businesses.json` (and `digest_email` in
`data/lead_config.json`) can be written as `"env:NAME"`; it is replaced at runtime by
the environment variable `NAME`. Add the address as a repo secret/variable, expose it in
the workflow `env:` block, and the committed file contains no personal data:

```json
{ "name": "Acme", "email": "env:ACME_OWNER_EMAIL" }
```

## Development

```bash
pip install -e ".[dev]"
python -m ruff check .      # lint
python -m mypy src          # types
python -m pytest -q         # ~130 tests incl. an end-to-end dry run of the daily pipeline
```

CI (`ci.yml`) runs the same three commands on every push and pull request. The test
suite is fully isolated: it uses temp databases/data dirs, forces `DRY_RUN`, and never
touches `data/` or `agent.db`. Optional: `pre-commit install` (ruff on commit, pytest on push).

## Operations notes

- **State:** JSON in `data/` is committed by the workflows; `agent.db` and `data/cache/`
  live in the Actions cache (saved even when a run fails). Weights are keyed by business
  *name*, so a rebuilt `agent.db` can never attach them to the wrong business.
- **Failures are loud:** pipelines exit non-zero when nothing was produced or nothing
  was delivered, and a `[pipeline] degraded run` issue is opened. `drift_check.yml`
  opens an issue if the LLM providers stop answering.
- **Feedback security:** set the repo variable `FEEDBACK_ALLOWED_USERS` (comma-separated
  GitHub logins) so only trusted people can move the weights through feedback issues.
- **Budgets:** `LEAD_TIME_BUDGET_SECONDS` caps lead scraping; Apify calls are made once
  per handle per run. Reels are best-effort: if ffmpeg/Piper fail the post falls back to a
  still image (the run warns instead of dropping the asset).
- **Scraping safety:** the website scraper refuses private/loopback/metadata addresses,
  and scraped text is fenced as untrusted data in every LLM prompt.

## Troubleshooting

| Symptom | Check |
|--------|-------|
| `All LLM providers failed` | Run the *Secrets Verify* workflow; check `GROQ_MODEL` / `GEMINI_MODEL` names are valid (unset secrets fall back to defaults) |
| Site not updating | *Settings → Pages* must use **Deploy from a branch: `gh-pages` / root**; check the run's *publish-site* step and its summary for the URLs |
| Emails not arriving | E-mail is off by default; set the repo variable `DELIVERY_MODE=both` and the `SMTP_*` secrets |
| Reels come out as images | See the run's warnings: usually `piper`/`ffmpeg` missing or the voice model download failed |
| Feedback buttons do nothing | The clicker needs a GitHub account that can open issues in this repo; the hourly *Process Feedback* job applies them |
| No new leads | Cooldown caches `data/lead_seen.json` (21 days) / `data/lead_rejected.json` (90 days); raise `target_count` or add niches |
