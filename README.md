# Self-Learning Business Content Intelligence Platform

Analyzes businesses (website + social handles), researches audience & competitors,
generates **viral-engineered content prompts** (scored 0-100, refined until ≥ threshold),
**renders them into real ad assets** — feed images with burned-in hooks and 9:16 Reel
MP4s with voiceover — and delivers them daily via email for a human to review and post
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
5. **Deliver**: saves `data/leads.json` + emails the digest with scores, reasons,
   rendered creatives, and "no contact email found" flags where applicable

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
  7. Email the finished assets via SMTP (Gmail) with 👍/👎/🔄/📎 feedback buttons

Sunday 8 AM IST (02:30 UTC) ──▶ weekly.yml
  Top creatives + real engagement metrics + why · audience shifts ·
  competitor moves · next week calendar

Hourly ──▶ feedback.yml
  Parse feedback issues → Bayesian weight updates → close issues
```

## How auto-learning works (no clicks required)

The system learns automatically from real post performance:

1. **Daily**: the business's own Instagram posts are scraped (Apify free tier)
2. **Matching**: posts linked via the email's **📎 "I posted this"** button are matched
   by exact post ID (no guessing). Unlinked posts fall back to date window
   (+5 days) + caption/prompt similarity (≥ 0.35 Jaccard)
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
| `SMTP_USER`, `SMTP_PASSWORD` | Gmail App Password (enable 2FA) |
| `SMTP_FROM`, `SMTP_HOST`, `SMTP_PORT` | Gmail SMTP (defaults set) |
| `APIFY_TOKEN` | Free tier social scraping (console.apify.com) |
| `HF_TOKEN` | Optional image fallback (Hugging Face FLUX.1-schnell) |
| `GITHUB_TOKEN` | Feedback issues + degraded-run tracking (built-in) |

LLM chain: Groq → Gemini → Pollinations (keyless last resort, loud warnings).
Images: Pollinations (keyless) → HF fallback. No local models run on CPU.
If any stage fails, a `[pipeline] degraded run — <date>` tracking issue is
created/updated so green CI never hides a dead run.

## Quick start

```bash
# Local dev
pip install -e .
python scripts/init_db.py
python scripts/onboard_business.py --name "Acme" --website "https://acme.com" --email "me@acme.com"

# Dry run locally: renders emails + assets to logs/preview/, sends nothing
DRY_RUN=true python -m src.scheduler.daily_job
DRY_RUN=true python -m src.scheduler.lead_job

# Trigger a run manually
gh workflow run daily.yml
gh workflow run leads.yml -f niches="bakery,home decor" -f target_count=5
```

Open `logs/preview/*.html` in a browser to inspect the actual rendered email;
images and MP4s land in `logs/assets/`.

## Repo layout

```
.github/workflows/   daily · weekly · feedback · onboard · leads (shared concurrency group)
assets/fonts/        Poppins Bold (OFL license) for hook overlays
src/fetch/           website scraper (incl. mailto extraction), social scraper
                     (Apify → instaloader), competitor finder, lead finder
                     (jittered search + cooldown cache)
src/intelligence/    audience mapper, trend radar, virality predictor, lead scorer
src/generation/      prompt forge (with reel_beats), virality gate, creative forge
src/delivery/        email sender (attachments + dry-run), github feedback links,
                     degraded-run tracker
src/learning/        bayesian updater, feedback processor (incl. POSTED links), auto-learner
src/scheduler/       daily_job, weekly_job, lead_job
src/storage/         SQLite models (agent.db lives in Actions cache, NOT git)
src/shared/          config, LLM client (Groq/Gemini/Pollinations), utils
templates/           daily_prompt.html, weekly_report.html, welcome.html, leads_digest.html
data/                businesses.json, lead_config.json, lead_seen.json, lead_rejected.json, weights.json (committed)
```

Rendered assets are uploaded as workflow artifacts and attached to emails — never committed to git.

## Onboarding a business

Use the `onboard.yml` workflow (Actions → Onboard Business → Run workflow) or run the
script locally. Demo seed records carry `"demo": true` and are skipped by scheduled runs.
Feedback buttons in emails open GitHub issues that the hourly feedback job turns into
Bayesian weight updates; the **📎 I posted this** button additionally links the published
post URL directly to the asset for exact engagement attribution.
