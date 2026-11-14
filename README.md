# Self-Learning Business Content Intelligence Platform

Analyzes businesses (website + social handles), researches audiences and competitors,
generates product-focused content prompts (scored 0-100), and renders static ads or vertical MP4 Reels.
Each run publishes previews to the **Content Hub** and queues the strongest passing creative through the existing
Buffer channel after automated media, claims, and duplicate checks (set the
repository variable `BUFFER_DRAFT_MODE=true` to force drafts; before queueing it skips known degraded assets and
unverified claims, checks for near-duplicates, and respects `BUFFER_QUEUE_LIMIT`, default 3, and `BUFFER_MIN_HOURS`, default 20 hours between auto-queued posts). Runs 100% in GitHub
Actions with open-weight generation models and open-source rendering tools; Hugging Face hosts model inference and Buffer handles publishing.
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
   → trends → virality gate) and renders finished image or Reel sample creatives
5. **Deliver**: saves `data/leads.json` + publishes the digest page (scores, reasons,
   rendered creatives, "no contact email found" flags where applicable)

Run weekly (Monday, `leads.yml`) or manually from the Actions tab. Configure niches,
follower range, target count, and digest email in `data/lead_config.json`.

## How it works

```
Mon/Wed/Fri 18:00 IST (12:30 UTC cron; GitHub may start scheduled runs hours late) ──▶ daily.yml
  0. AUTO-LEARN: scrape the business's own posts → match to previously sent
     prompts (exact post ID when the user tapped 📎 "I posted this", else
     date + caption similarity) → score real engagement → update Bayesian weights
  1. Scrape website profile (httpx)
  2. Fetch social data + competitor posts (Apify free tier / instaloader)
  3. Audience Mapper → personas, pain points, themes
  4. Trend Radar → today's topics, hashtags, gaps, viral patterns
  4b. Campaign Strategy: objective, persona, customer problem, emotional trigger, CTA and KPIs; three angles
     rated on a fixed rubric, winner chosen in code (see docs/agency-workflow.md; CAMPAIGN_STRATEGY_ENABLED=false disables)
  5. Virality Gate: generate → predict → if score < 75, refine with hints → repeat
  6. Creative Forge → render single images or assemble vertical MP4 Reels from scene beats with FFmpeg
  7. Publish the finished assets to the Content Hub (GitHub Pages) with 👍/👎/🔄/📎 feedback buttons

Sunday 8 AM IST (02:30 UTC) ──▶ weekly.yml
  Top creatives + real engagement metrics + why · audience shifts ·
  competitor moves · next week calendar

Every 6h ──▶ engagement.yml  (scrape own posts → auto-learn)

Every 3 hours and on each feedback issue ──▶ feedback.yml
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
   "bad" otherwise. Posts younger than 48 h are not judged yet (`MATURITY_HOURS`), and
   posts whose like counts are hidden are skipped, so immature numbers never teach the system
5. **Update**: outcomes feed the Bayesian weight loop (`data/weights.json`, keyed by business
   slug) across prompt type × format × audience segment, so the next round favors what worked.
   Each prompt is counted once and re-judged for 30 days as its numbers mature

What this is, precisely: *online count-based (Beta-Bernoulli) learning* with a rule-based
benchmark and a bounded ranking boost (±6 points). It is **not** model training, and the
"history" given to the LLM is retrieval, not learning. Every update is appended to
`data/learning_ledger.jsonl`, so any weight can be explained, replayed or rolled back
(`python scripts/rollback_weights.py`) and measured (`python scripts/evaluate_learning.py`).

Manual 👍/👎/🔄 clicks remain as additional signals. Run the auto-learner standalone
with `python scripts/run_auto_learn.py`.

## Brand facts (fill this in)

The generator only knows what it is told. A website scrape of an Instagram URL returns almost nothing, so without help it
knows only the business name and invents services, results and turnaround times to fill the gap. `data/brand_facts.json`
is where you write the **true** facts: `offers`, `audience`, `voice`, `proof_points`, `allowed_claims`, `do_not_say`.
Everything in it is treated as verified: the generator may state it, and numbers that appear in it pass the claims guard
(see *Claims guard*). Until `offers` or `proof_points` has content the run logs a warning and posts stay generic.

## Claims guard

Variants whose hook, title or caption contain a percentage, multiplier, money amount, client count, delivery time or
number of deliverables that is not in the business data are dropped before scoring, and the publish step re-checks every
candidate. Creatives that fell back to the flat branded card, or were already posted (`data/published_log.json`), are
never picked. The guard catches numeric claims only; vague comparative wording still needs your review.

## Learning status

The learning loop is real but starved: it updates only from matched, mature posts with measurable engagement. Until a
meaningful number of posts has been matched (check `data/learning_ledger.jsonl`), the weights are near their prior and
barely influence ranking. Treat "self-learning" as the mechanism, not as evidence that it has learned anything yet.

## Requirements (GitHub Secrets)

| Secret | Purpose |
|--------|---------|
| `GROQ_API_KEY` | Hosted inference for the open-weight GPT-OSS model |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | Hosted inference for the open-weight FLUX.1-schnell image model |
| `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_HOST`, `SMTP_PORT` | Only for the legacy e-mail mode (`DELIVERY_MODE=email\|both`) |
| `SITE_SECRET` | Optional: keyed page URLs and an unlisted index. Obscurity only, not access control (see *Content Hub → Privacy*) |
| `DELIVERY_MODE` (repo *variable*) | `site` (default), `email` or `both` |
| `APIFY_TOKEN` | Free tier social scraping (console.apify.com) |
| `GITHUB_TOKEN` | Feedback issues + degraded-run tracking (built-in) |
| `FEEDBACK_SIGNING_KEY` | Signs the 👍/👎 links. **Required for feedback to be accepted** (without it links are inert) |
| `STATE_ENCRYPTION_KEY` | Encrypts the private Actions state cache; generated and configured for this repository |
| `FEEDBACK_ALLOWED_USERS` (repo *variable*) | Comma-separated GitHub logins whose feedback issues count. **Required**: unset = feedback is ignored (fails closed) |
| `OWNER_EMAIL`, `DIGEST_EMAIL` | Referenced from `data/*.json` as `env:OWNER_EMAIL` / `env:DIGEST_EMAIL`, so no address is committed |
| `BELOW_THRESHOLD_POLICY` (repo *variable*) | `flag` (default: publish with a "review" badge) or `block` (publish nothing under the bar) |
| `LLM_KEYLESS_FALLBACK` (repo *variable*) | `false` by default. `true` lets prompts go to the public keyless endpoint as a last resort |

LLM chain: every configured provider in `LLM_PROVIDER_ORDER` (see *Free fallback chain*). If all fail the run is
reported as degraded. The keyless text fallback is **off** unless `LLM_KEYLESS_FALLBACK=true`.
Images: Cloudflare FLUX → Gemini/HF/Together → Pollinations (keyless, last: it is rate limited on shared runners) → local branded card. No local models run on CPU.
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
images land in `logs/assets/`.

## Repo layout

```
.github/workflows/   daily · weekly · feedback · engagement · onboard · leads (shared concurrency group) · ci · drift_check
assets/fonts/        Poppins Bold (OFL license) for hook overlays
src/fetch/           website scraper (incl. mailto extraction), social scraper
                     (Apify → instaloader), competitor finder, lead finder
                     (jittered search + cooldown cache)
src/intelligence/    audience mapper, trend radar, virality predictor, lead scorer
src/generation/      prompt forge, virality gate, creative forge
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
Feedback buttons on the site open GitHub issues that the feedback job (every 3 hours, and immediately when an issue is opened or edited) turns into
Bayesian weight updates; the **📎 I posted this** button additionally links the published
post URL directly to the asset for exact engagement attribution. Posting alone is
not counted as a success: the learner scores that post's real engagement later.

## Content Hub (the static site)

E-mail is replaced by a small static website published to GitHub Pages
(`https://<owner>.github.io/<repo>/`, or the URLs printed in each run's summary).

- **What you get per business:** every prompt with its finished single image (download
  buttons), the copy-paste caption, hashtags, best time to post, first comment, the full
  creative brief, why it should work (strengths/risks), the 👍/👎/🔄/📎 feedback buttons,
  a local "posted" tracker, search and type filters. Plus weekly deep-dives, an
  "about this setup" panel and a banner whenever a day's generation failed.
- **Leads:** the weekly digest has its own page.
- **How it is built:** each pipeline restores the `gh-pages` branch into `site/`, adds
  its entries (images are re-encoded to web JPEGs), prunes
  entries older than `SITE_RETENTION_DAYS` (45) and force-pushes a single orphan commit.
  The JSON manifests in `site/data/` are the source of truth; the HTML is regenerated.
- **Hosting (pick one; the `gh-pages` branch is just a folder of static files):**
  1. **GitHub Pages** — *Settings → Pages → Deploy from a branch → `gh-pages` / `/ (root)`*
     (needs repo admin; on a **private** repo it also needs a paid GitHub plan).
  2. **Free for private repos:** connect **Cloudflare Pages** (or Netlify / Vercel) to the
     repo, production branch `gh-pages`, *no build command*, output directory `/`.
  The first publish creates the branch; every later publish redeploys automatically.
- **Privacy: treat everything on the hub as public.** While the repository is public, so are the `gh-pages` branch
  and `data/site_manifests/` on `main` (they contain every prompt, caption and strategy, and the keyed page path).
  `SITE_SECRET` only hides pages from casual browsing and the index; it is **not** access control and is
  not private while the repo is public. Contact e-mail addresses and phone numbers are never published and pages
  are `noindex`. Never put anything in a brand fact, caption or lead note that you would not publish. For genuinely
  private content make the repository private (and host the site on a plan or service that supports it).
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
| **Text** (`LLM_PROVIDER_ORDER`) | Groq inference with OpenAI GPT-OSS 120B (Apache-2.0) | Scheduled GitHub Actions use this open-weight model only. |
| **Images** (`IMAGE_PROVIDER_ORDER`) | Cloudflare Workers AI with FLUX.1-schnell (Apache-2.0), then local Pillow fallback | Scheduled GitHub Actions use this open-weight model only. The local fallback is flagged by automated publication QA and will not be queued. |
| **Creative** | Static image posts and FFmpeg-rendered vertical MP4 Reels | The prompt generator supports both formats. Reel videos are hosted with the public site assets and scheduled through the configured Buffer channel. |

The scheduled workflow restricts generation to Apache-2.0 open-weight models and does not
use Gemini or other closed-model providers. Groq and Cloudflare host inference (usage may
be billable); the rendering tools (FFmpeg, Pillow, and Piper) are open source. The daily
*Drift Check* probes configured providers and opens an issue when generation services fail.

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
python -m pytest -q         # ~600 tests incl. an end-to-end dry run of the daily pipeline (CI also enforces >= 80% coverage)
python scripts/benchmark_pipeline.py   # reproducible timing with simulated latency
```

CI (`ci.yml`) runs the same three commands on every push and pull request. The test
suite is fully isolated: it uses temp databases/data dirs, forces `DRY_RUN`, and never
touches `data/` or `agent.db`. Optional: `pre-commit install` (ruff on commit, pytest on push).

## Operations notes

- **State:** JSON in `data/` is committed by the workflows; `agent.db` and `data/cache/`
  live in an encrypted Actions cache (saved even when a run fails). Configure the repository
  secret `STATE_ENCRYPTION_KEY` (a long random value, e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`);
  `FEEDBACK_SIGNING_KEY` is used as a fallback. Cached state is encrypted before upload, so
  pull request workflows cannot read it. Weights are keyed by business
  *slug* (`Business.weights_key`), so a rebuilt `agent.db` can never attach them to the wrong business.
- **Failures are loud:** pipelines exit non-zero when nothing was produced or nothing
  was delivered, and a `[pipeline] degraded run` issue is opened. `drift_check.yml`
  opens an issue if the LLM providers stop answering.
- **Feedback security:** feedback issues are accepted only from `FEEDBACK_ALLOWED_USERS`, carry an
  HMAC token (`FEEDBACK_SIGNING_KEY`) bound to one prompt, and count once per prompt and button.
  With either setting missing the job refuses to apply anything.
- **Speed:** LLM calls (2), image renders (3) and scrapes (2) run concurrently within those caps
  (`LLM_CONCURRENCY`, `IMAGE_CONCURRENCY`, `FETCH_CONCURRENCY`; `1` restores sequential behaviour).
  Publishing re-renders only the business that changed.
- **Budgets:** `LEAD_TIME_BUDGET_SECONDS` caps lead scraping; Apify calls are made once
  per handle per run. Each prompt renders according to its normalized image or Reel format.
- **Scraping safety:** the website scraper refuses private/loopback/metadata addresses,
  and scraped text is fenced as untrusted data in every LLM prompt.

## Troubleshooting

| Symptom | Check |
|--------|-------|
| `All LLM providers failed` | Run the *Secrets Verify* workflow; check `GROQ_MODEL` / `GEMINI_MODEL` names are valid (unset secrets fall back to defaults) |
| Site not updating | *Settings → Pages* must use **Deploy from a branch: `gh-pages` / root**; check the run's *publish-site* step and its summary for the URLs |
| Emails not arriving | E-mail is off by default; set the repo variable `DELIVERY_MODE=both` and the `SMTP_*` secrets |
| Feedback buttons do nothing | The clicker needs a GitHub account that can open issues in this repo; the *Process Feedback* job (every 3 hours, or on issue events) applies them |
| No new leads | Cooldown caches `data/lead_seen.json` (21 days) / `data/lead_rejected.json` (90 days); raise `target_count` or add niches |
