# Weekly Analysis Prompt

Used by `WeeklyPipeline` to build the Sunday deep-dive report email.

## Inputs

- Business config (name, handles, threshold)
- Prompts sent this week (text, predicted virality, user feedback)
- Learned Bayesian weights (topic/format/segment → weight)

## Output schema

```json
{
  "top_performing": [
    {"prompt": "...", "score": 0, "feedback": "good|bad|none", "why": "..."}
  ],
  "audience_insights": ["..."],
  "competitor_moves": ["..."],
  "content_pillars": [
    {"pillar": "...", "rationale": "...", "suggested_frequency": "..."}
  ],
  "calendar": ["Mon: ..."],
  "recommendations": ["..."],
  "risk_alerts": ["..."]
}
```

## Usage

Rendered via `templates/weekly_report.html` and sent with
`EmailSender.send_weekly_report`. Failed generations fall back to a
minimal report so the email always goes out.