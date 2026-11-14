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
- **The Content Hub is public.** `SITE_SECRET` only makes page URLs hard to guess; it is **not** access control.
  While the repository is public, the `gh-pages` branch and `data/site_manifests/` (which contain the keyed page path)
  are public too, so anyone can read every prompt and caption. Make the repository private if that is not acceptable.
- **Feedback links** are HMAC-signed (`FEEDBACK_SIGNING_KEY`) and only accepted from the users listed
  in `FEEDBACK_ALLOWED_USERS`. With either unset, feedback processing refuses to run.
- **Scraping** only fetches public http(s) hosts; private, loopback and link-local addresses are refused.

## Operational rules

- Rotate any key that has ever appeared in a log, issue or commit.
- The Content Hub is public while the repository is public; make the repository private if it must not be.
- Dependencies are scanned with `pip-audit` in CI and updated by Dependabot.
