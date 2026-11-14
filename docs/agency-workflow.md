# From content automation to an advertising agency workflow

Status: increments 1-3 shipped (strategy stage, explainable lead scoring, research provenance; measured competitor
benchmarks, evidence-based posting window, per-lead recommended campaign). Everything else below is the prioritised roadmap. Written 2026-11-14 from a read of the actual code, not from the README.

## 1. What the system does today

```
Weekly   leads.yml        search (DuckDuckGo/Bing + Apify) -> enrich site -> score -> top N -> sample creatives -> digest page
3x/week  daily.yml        auto-learn -> scrape site -> own posts + LLM-suggested competitors (scraped)
                          -> audience map (LLM) -> trend scan (LLM knowledge)
                          -> [NEW] campaign strategy (objective, persona, problem, trigger, 3 rubric-scored angles)
                          -> virality gate (generate <=5 variants, LLM score 0-100, refine, claims guard)
                          -> render (FLUX stills / Piper+FFmpeg Reels) -> hub page -> Buffer draft/queue
6-hourly engagement.yml   Buffer per-post metrics (exact IDs) + Instagram scrape fallback -> Beta-Bernoulli weights
Weekly   weekly.yml       report
```

## 2. Agency capability map

| Agency function | Today | Verdict |
| --- | --- | --- |
| Lead discovery | Keyword web search + Apify Instagram search per niche and city | Works, but yield is low and sourcing is thin (no directories, maps, hashtag or competitor-follower mining) |
| Lead scoring | **Now:** 7 explainable dimensions (engagement gap, audience size, activity, buying intent, content potential, reachability, competition) with notes, `why_target`, data gaps | Improved. Still no demand, pain-point or conversion-probability model: no data to fit one |
| Market research | Website scrape, own captions, competitors suggested by an LLM then scraped, trends from LLM memory | Weak. **Now** unverified competitors are excluded, trends are labelled as hypotheses, and verified competitors are benchmarked from their posts (cadence, Reel share, CTA use, hashtags, best openers). No live trend/search data |
| Audience research | LLM audience map from profile + captions | Reasonable hypothesis generator; never validated against real audience data |
| Strategy | **Now:** one decision per run: objective, persona, problem, emotional trigger, differentiator, CTA, KPIs, test hypothesis; 3 angles rated on a rubric, winner chosen in code with a bounded learned-weight nudge | New. Was entirely missing (the forge went straight from context to 5 variants) |
| Creative development | Forge writes 5 variants (briefs, captions, hooks, Reel beats and narration); virality gate refines | Good craft rules (hook psychology, claims guard). Variants now serve one brief |
| Production | FLUX stills, Piper voice + FFmpeg Reels, local fallback card, QA flags | Working; Reel quality gates exist. No music/sound design, no B-roll or brand kit beyond a font |
| Review | Hub page + Buffer drafts; `promote_drafts` auto-promotes after the review window | Review exists. No in-hub approve/reject, no creative scorecard for the reviewer |
| Publishing | Buffer `automatic` queue slot. **Now** the recommended window is measured from scraped post performance (3-hour blocks, per-account normalised, with confidence and evidence on the hub), falling back to a labelled estimate | Better. Still not sent to Buffer as a custom schedule |
| Measurement | Buffer metrics (reach, reactions, comments, shares, saves, follows, watch time) when mature | Right design, but starved: 0 matured posts so far. No clicks, leads, conversions, CPL or ROAS |
| Learning | Bayesian weights per topic, format, audience, hook; one exploration variant 25% of runs | Sound, bounded. Needs data. Experiments are random, not designed |
| Operations | Failure alerts, degraded-run issues, token budgeting, claims guard | Good |

## 3. Proposed agency workflow (target)

```
DISCOVER  leads + trends + competitor moves          (sources, not LLM memory)
SELECT    score on evidence; explain; reject the rest  <- lead scorer v2 (done)
RESEARCH  site, offers, USP, audience, competitors     <- provenance (done); live signals (roadmap)
STRATEGISE objective, persona, problem, trigger, angles, KPI, hypothesis   <- strategist (done)
CREATE    concepts that execute the chosen angle; scripts, scenes, VO, overlays, captions
REVIEW    scorecard + approve/reject; claims and brand checks
PUBLISH   platform, format, time from data; queue or draft
MEASURE   reach, watch time, saves, shares, follows, clicks, leads
LEARN     update hook/angle/format/time weights; designed experiments; roll back regressions
```

## 4. Increment 1 (shipped)

| Change | Files | Value | Risk and control |
| --- | --- | --- | --- |
| Campaign strategy stage | `src/intelligence/campaign_strategist.py`, `daily_job.py`, `prompt_forge.py`, `virality_gate.py` | Every run answers WHAT, WHO, WHY before creating. Winner is computed from a fixed rubric, so decisions are reproducible and explainable | One more LLM call (~2.5k tokens). Failure returns no brief and the old flow runs unchanged. Kill switch `CAMPAIGN_STRATEGY_ENABLED=false` |
| Strategy shown to the reviewer | `site_publisher.py`, `business.html` | Reviewer sees objective, angle, rationale, alternatives, CTA, KPIs and the test hypothesis | Public page: contact data redacted as before |
| Explainable lead scoring | `lead_scorer.py`, `lead_job.py`, digest page | Selection explains itself; missing data is listed, never invented | Scores shift slightly; `min_opportunity_score` (60) still applies |
| Research provenance | `research.py`, `audience_mapper.py`, `trend_radar.py` | Stops hallucinated competitors from being treated as evidence | None |

## 4b. Increment 2 (shipped)

| Change | Files | Value | Risk and control |
| --- | --- | --- | --- |
| Competitor benchmarks from scraped posts | `competitor_analysis.py`, `campaign_strategist.py` | The strategist receives measured facts (cadence, Reel share, CTA share, hashtags) instead of raw captions; says so explicitly when there is no verified data | Needs Apify; verified competitors only |
| Evidence-based posting window | `posting_time.py`, `daily_job.py`, hub | Time chosen by relative engagement per 3-hour block, not by posting habit; shows confidence and basis | Needs >= 8 usable posts, otherwise falls back with a clear label |
| Per-lead recommended campaign | `lead_job.py`, digest page | Sales-ready: each top lead shows objective, persona, problem, angle and CTA | One extra LLM call per top lead (`LEAD_STRATEGY_COUNT`, default 3) |

## 4c. Increment 3 (shipped): hands-off operation

| Change | Files | Value | Risk and control |
| --- | --- | --- | --- |
| **Unattended draft review and scheduling** | `draft_review.py`, `slot_planner.py`, `scripts/promote_drafts.py`, `promote_drafts.yml` | A draft nobody reviewed is re-checked after the review window (quality score, unverified claims, duplicates, media) and scheduled at an exact time from the measured window or an experiment arm, 20h+ clear of other posts. Failures are HELD with a reason in the Actions summary | Never publishes below-bar, unscored or claim-carrying drafts. Falls back to the automatic queue slot if Buffer rejects a custom time. `DRY_RUN` input reports without changing Buffer |
| Custom schedule for new posts | `buffer_publisher.py`, `publish_buffer.py` | Non-draft posts publish at the planned slot instead of Buffer's next automatic slot | Falls back to the queue on any error |
| **Designed experiments** | `experiments.py`, daily and engagement workflows | One variable per ISO week (hook, posting block, format), control vs challenger, balanced from the published log; verdict only with >= 4 matured posts per arm and |t| >= 2 | Never promotes a creative under the quality bar. Winners only inform strategy hints and a preferred posting hour |
| **Competitor discovery by Instagram search** | `competitor_finder.py` | Real accounts (search results) in the right size band, scraped in waves until 3 are verified | Apify credits; LLM suggestions remain as a supplement |
| **Live trend signals** | `trend_sources.py`, `trend_radar.py` | Dated, attributed Google News / Hacker News / hashtag-post signals; topics marked `live_signal` or `model_knowledge` | Each source fails independently; hashtag source needs Apify |
| **Conversion tracking** | `conversions.py`, `process_feedback.py`, hub, weekly report | A signed "Got a lead?" link per post logs the lead against the exact creative, angle, hook and format; weekly "Results that matter" | Owner-reported: Instagram DMs cannot be detected without Meta Graph API access |

## 5. Roadmap

Priority = value / effort given today's constraints (free-tier LLM at 8k tokens/min, one active account, learning starved).

| P | Improvement | Business value | Needs | Effort |
| --- | --- | --- | --- | --- |
| P1 | **Automatic conversion detection**: Meta Graph API (profile visits, website taps, follows per media) and DM counts replace owner-reported leads | Fully automatic lead attribution | Instagram Business account, Meta app review, long-lived token | High |
| P2 | **More live sources**: Google Trends, Reddit and YouTube search | Broader market signal | API keys / rate limits | Medium |
| P2 | **Reviewer scorecard + approve/reject in the hub** (writes a GitHub issue or Buffer action) | Faster, safer review; feeds learning | GitHub issue flow exists | Medium |
| P3 | Multi-platform formats (YouTube Shorts, Facebook): format adapters after the Instagram loop works | Reach | Platform APIs via Buffer | High |
| P3 | Music and sound design, brand kit (palette, logo, fonts) per business | Production polish | Licensed audio, brand assets | Medium |
| P3 | Paid amplification (boost winners) and ROAS | Scale | Ad account access, budget, approval | High |

## 6. Dependencies and risks

- **LLM budget:** Groq free tier allows 8,000 tokens per minute. Every extra LLM stage competes with generation; per-lead strategies must be queued or the tier raised.
- **Truthfulness:** the claims guard blocks numeric and proof claims not present in `data/brand_facts.json`. Fill `proof_points` or posts stay generic.
- **Learning data:** nothing can be optimised without matured posts; until ~20 exist, treat "self-learning" as a mechanism, not evidence.
- **Platform terms:** scraping Instagram (Apify, instaloader) may breach its terms; prefer Buffer metrics and official APIs where possible.
- **Publishing control:** keep Buffer drafts and the review window until measured quality justifies autonomy.

## 7. How to verify increment 1

1. `pytest` (strategy, scoring and provenance tests are in `tests/test_campaign_strategist.py`, `test_lead_scoring_v2.py`, `test_research_provenance.py`).
2. Run the Daily workflow: the log shows `Campaign strategy for <business>: Angle '...' scored N/100 ...` and `Research quality for ...`; each hub entry has a "Campaign strategy" panel.
3. Run Leads: each digest lead shows "Why target", a score breakdown and "Not measured" gaps.
