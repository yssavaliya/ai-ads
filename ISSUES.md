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
| Successful scheduled workflows spawned a separate skipped “Alert on Pipeline Failure” run | Fixed: removed the `workflow_run` watcher and moved issue creation into failure-only steps in each scheduled workflow. Successful runs no longer create alert runs. |
| Four workflows passed the encryption key to `actions/checkout`, which ignored the unsupported input | Fixed: the key is now passed to `setup-agent`; added a workflow check to prevent future miswiring. |

## Remaining external verification

| Issue | Status |
| --- | --- |
| Plaintext cache entries created before encryption may still be available to pull-request workflows | Needs purge in GitHub Actions cache. This environment has an invalid `gh` credential and cannot reach the GitHub API, so existing cache entries could not be enumerated or deleted. |
| End-to-end daily workflow execution | Verified: [Actions run 40026910360](https://github.com/yssavaliya/ai-ads/actions/runs/40026910360) completed successfully, including pipeline generation, artifact upload, state persistence step, and site publish. The generated artifact contained only still-image files (`.png`/`.jpg`). State persistence warned that neither encryption key was configured, so private state was not cached. |
| Alert behavior after removing the success-triggered watcher | Verified: the no-pending-feedback run [40028850918](https://github.com/yssavaliya/ai-ads/actions/runs/40028850918) succeeded with the inline alert step skipped, and no separate alert run was created. Inline alerts run only after a job starts; GitHub runner or job-start failures cannot be reported from inside that job. |

## Verification limits

| Check | Status |
| --- | --- |
| Full local test suite after the single-image change | Blocked in this shell: pytest cannot create its temp directory under the configured Windows profile or workspace (`PermissionError`). A direct hostile-response forge check and Python compilation passed. The full suite passed earlier in this review before this change (507 passed, 3 skipped). |

## Current required configuration

Set repository secret `STATE_ENCRYPTION_KEY`; the workflows fall back to `FEEDBACK_SIGNING_KEY` if that secret is absent. Neither key was available in the verified Actions run, so private state was not cached. The secret must be the same across runs so old encrypted state can be restored.
