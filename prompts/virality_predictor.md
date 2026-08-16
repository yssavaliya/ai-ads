# Virality Predictor Prompt

The virality predictor is an LLM-based scorer used by `ViralityGate`. It evaluates
a generated content prompt and returns a structured JSON verdict.

## Inputs

- Business profile (products, positioning, tone)
- Audience map (personas, pain points, content themes)
- Today's trend data (topics, hashtags, gaps, viral patterns)
- Prompt type (educational, storytelling, social_proof, ...)
- The content prompt text to evaluate

## Output schema

```json
{
  "virality_score": 0,
  "virality_label": "LOW|MEDIUM|HIGH",
  "virality_probability": 0.0,
  "key_factors": ["..."],
  "risks": ["..."],
  "reasoning": "...",
  "improvement_hints": ["..."]
}
```

## Scoring rubric

The prompt instructs the model to act as a viral content strategist for the
Indian market and evaluate:

- Hook strength (first 3 words)
- Emotional trigger
- Specificity (numbers, concrete examples)
- Indian cultural relevance
- Platform fit (Instagram visual-first, LinkedIn thought-leadership)
- Scroll-stopping potential
- Shareability
- Fit with audience map

## Usage

`improvement_hints` are fed back into `PromptForge.refine_with_hints` so the
virality gate can iterate until `virality_score >= threshold` (default 75).