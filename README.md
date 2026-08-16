# Self-Learning Business Content Intelligence Platform

Analyzes businesses (website + social handles), researches audience & competitors,
generates **viral-engineered content prompts** (scored 0-100, refined until ≥ threshold),
and delivers them daily via email — **no auto-posting**. Runs 100% in GitHub Actions
with free & open-source tooling only.

## How it works

```
23:59 IST daily (18:29 UTC) ──▶ daily.yml
  0. AUTO-LEARN: scrape the business's own posts → match to previously sent
     prompts (date + caption similarity) → score real engagement → update
     Bayesian weights (no clicks needed)
  1. Scrape website profile (Playwright/httpx)
  2. Fetch social data + competitor posts (Apify free tier / Playwright)
  3. Audience Mapper → personas, pain points, themes
  4. Trend Radar → today's topics, hashtags, gaps, viral patterns
  5. Virality Gate: generate → predict → if score < 75, refine with hints → repeat
  6. Email 3 prompts via SMTP (Gmail) with 👍/👎/🔄 feedback buttons (GitHub Issues)

Sunday 8 AM IST (02:30 UTC) ──▶ weekly.yml
  Top prompts + real engagement metrics + why · audience shifts ·
  competitor moves · next week calendar

Hourly ──▶ feedback.yml
  Parse feedback issues → Bayesian weight updates → close issues
```

## How auto-learning works (no clicks required)

The system learns automatically from real post performance:

1. **Daily**: the business's own Instagram/LinkedIn posts are scraped (Apify free tier)
2. **Matching**: each post is matched to a previously sent prompt by post date
   (posted within 5 days of the prompt) + caption/prompt token similarity
3. **Scoring**: engagement rate = (likes + 2·comments + 3·shares + 2·saves) / reach
4. **Benchmark**: a post is "good" if its rate ≥ the business's own 30-day median,
   "bad" otherwise
5. **Update**: good/bad outcomes feed the same Bayesian weight loop as manual
   clicks (`data/weights.json`), so tomorrow's prompts favor what actually worked

Manual 👍/👎/🔄 clicks in emails remain as an additional, stronger signal.
Run the auto-learner standalone with `python scripts/run_auto_learn.py`.

## Requirements (GitHub Secrets)

| Secret | Purpose |
|--------|---------|
| `SMTP_USER`, `SMTP_PASSWORD` | Gmail App Password (enable 2FA) |
| `SMTP_FROM`, `SMTP_HOST`, `SMTP_PORT` | Gmail SMTP (defaults set) |
| `GROQ_API_KEY`, `GROQ_MODEL` | Free LLM fallback (console.groq.com) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Free LLM fallback (aistudio.google.com) |
| `APIFY_TOKEN` | Free tier social scraping (console.apify.com) |
| `GITHUB_TOKEN` | For feedback issues (defaults to GITHUB_TOKEN) |

Primary LLM is the **OpenCode CLI** (installed in workflows, free gateway).
No local models run on CPU.

## Quick start

```bash
# Local dev
pip install -e .
python scripts/init_db.py
python scripts/onboard_business.py --name "Acme" --website "https://acme.com" --email "me@acme.com"

# Trigger a run manually
gh workflow run daily.yml
```

## Repo layout

```
.github/workflows/   daily · weekly · feedback · onboard
src/fetch/           website scraper, social scraper, competitor finder
src/intelligence/    audience mapper, trend radar, virality predictor
src/generation/      prompt forge, virality gate
src/delivery/        email sender (SMTP), github feedback links
src/learning/        bayesian updater, feedback processor
src/scheduler/       daily_job, weekly_job
src/storage/         SQLite models (agent.db committed each run)
src/shared/          config, LLM client (opencode + free APIs), utils
templates/           daily_prompt.html, weekly_report.html, welcome.html
prompts/             LLM prompt reference docs
data/                businesses.json, weights.json (committed)
```

## Onboarding a business

Use the `onboard.yml` workflow (Actions → Onboard Business → Run workflow) or
run the script locally. Feedback buttons in emails open GitHub issues that the
hourly feedback job turns into Bayesian weight updates.