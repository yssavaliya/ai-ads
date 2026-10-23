# Security policy

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private vulnerability reporting
(*Security -> Report a vulnerability*) on this repository, or contact the repository owner directly.
Include what you found, how to reproduce it and what you think the impact is.

## What this project handles

- **Third-party API keys** (LLM, image, Apify) live only in GitHub Actions secrets; they are never
  written to the repository, logs or the published site.
- **Business contact data** (e-mails, phone numbers) is redacted before any text is sent to an LLM
  and is never published to the Content Hub. `data/*.json` hold `env:NAME` references, not addresses.
- **The Content Hub** is protected by an unguessable URL (`SITE_SECRET`), **not** by access control.
  Anyone with a link can read that business's page. Rotate `SITE_SECRET` to invalidate every link.
- **Feedback links** are HMAC-signed (`FEEDBACK_SIGNING_KEY`) and only accepted from the users listed
  in `FEEDBACK_ALLOWED_USERS`. With either unset, feedback processing refuses to run.
- **Scraping** only fetches public http(s) hosts; private, loopback and link-local addresses are refused.

## Operational rules

- Rotate any key that has ever appeared in a log, issue or commit.
- Keep the repository private unless the Content Hub is meant to be public.
- Dependencies are scanned with `pip-audit` in CI and updated by Dependabot.
