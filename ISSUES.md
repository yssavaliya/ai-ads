# Project issue tracker

Updated 2026-10-24 during iterative code and workflow review.

## Resolved in this review

| Issue | Status |
| --- | --- |
| Public static-site JSON manifests bypassed unlisted URLs | Fixed: manifests are stored under `data/site_manifests/`; legacy copies are migrated out of the Pages tree. They are included in state commits so later jobs can rebuild the site. |
| Pull request workflows could restore plaintext business data from Actions cache | Fixed in current workflows: SQLite/cache data is archived and Fernet-encrypted before cache upload; restore decrypts only when a key is available. |
| Engagement history could mix businesses | Fixed: new log entries carry a stable business key and predictors filter by it. |
| Daily dates could use the runner's UTC date instead of the configured business timezone | Fixed. |
| Feedback processing fetched only the first page of GitHub issues | Fixed: follows pagination links. |
| Lead scraping budget started after learning/discovery/enrichment | Fixed: budget begins at pipeline start and wraps these phases. |
| Posts with hidden likes or unknown/immature timestamps could contaminate learning | Fixed: unusable and immature posts are excluded. |
| Website-supplied URLs could use unsafe schemes | Fixed: published links must be HTTP(S) URLs with a hostname. |
| Predictor history could prefer the oldest engagement records | Fixed: matching business records are traversed newest-first. |
| Repeated regenerate feedback could duplicate weight updates/prompts | Fixed: replacement linkage is persisted and negative feedback is applied once. |
| Failed replacement delivery could close feedback as if successful | Fixed: issue stays open for retry and the draft prompt is removed. |
| Generated content could ask for a Reel or carousel instead of one image | Fixed: the forge requests one static image, normalizes every model response and refinement to `single image`, clears Reel beats, and adds a one-frame image requirement. The virality gate applies the same rule to alternate forge implementations. |
| Trend analysis and setup still advertised or provisioned video output despite the single-image product requirement | Fixed: trend radar now only recommends `single image`, user-facing copy describes one still image, and scheduled/CI workflows no longer install Piper/FFmpeg video dependencies. Added a regression test for hostile trend output and workflow configuration. |
| Successful scheduled workflows spawned a separate skipped “Alert on Pipeline Failure” run | Fixed: removed the `workflow_run` watcher and moved issue creation into failure-only steps in each scheduled workflow. Successful runs no longer create alert runs. |
| Four workflows passed the encryption key to `actions/checkout`, which ignored the unsupported input | Fixed: the key is now passed to `setup-agent`; added a workflow check to prevent future miswiring. |
| Private state cache encryption secret was not configured | Fixed: generated and set repository secret `STATE_ENCRYPTION_KEY`; confirmed it appears in GitHub's secret metadata without reading its value. |
| Legacy private database caches were stored without encryption | Fixed: removed all 81 `agent-db-*` cache entries; other cache namespaces were retained. The database cache starts fresh under the encrypted cache namespace; committed weights and learning ledger remain available. |

## Remaining external verification

| Issue | Status |
| --- | --- |
| New encrypted state-cache save and restore | Needs an Actions run after the new `STATE_ENCRYPTION_KEY` was set; existing verified runs predate it. |
| End-to-end daily workflow execution | Verified: [Actions run 40026910360](https://github.com/yssavaliya/ai-ads/actions/runs/40026910360) completed successfully, including pipeline generation, artifact upload, state persistence step, and site publish. The generated artifact contained only still-image files (`.png`/`.jpg`). State persistence warned that neither encryption key was configured, so private state was not cached. |
| Alert behavior after removing the success-triggered watcher | Verified: no-pending-feedback runs [40028850918](https://github.com/yssavaliya/ai-ads/actions/runs/40028850918) and [40029267748](https://github.com/yssavaliya/ai-ads/actions/runs/40029267748) succeeded with no separate alert run; the latter also had no ignored encryption-key inputs. CI [40029008201](https://github.com/yssavaliya/ai-ads/actions/runs/40029008201) passed after the workflow corrections. Inline alerts run only after a job starts; GitHub runner or job-start failures cannot be reported from inside that job. |

## Verification limits

| Check | Status |
| --- | --- |
| Targeted prompt tests | Passed outside the sandbox: 10 passed. The sandboxed test invocation cannot create pytest's Windows temp directory (`PermissionError`); run full suite outside sandbox before closing. |

## Current required configuration

Repository secret `STATE_ENCRYPTION_KEY` is configured. It must remain stable across runs so encrypted state can be restored. Existing unencrypted database cache entries were purged; the encrypted database state will be rebuilt by subsequent jobs.
