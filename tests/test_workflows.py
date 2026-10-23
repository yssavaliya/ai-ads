"""Structural checks for the GitHub Actions setup.

Every one of these encodes a failure seen in the real run history: green runs
that produced nothing (swallowed exit codes), a drift check that could never
alert, state files that were never committed, and shell injection via inputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

GH = Path(__file__).resolve().parents[1] / ".github"
WORKFLOWS = sorted((GH / "workflows").glob("*.yml"))
PIPELINE_WORKFLOWS = {
    "daily": "src.scheduler.daily_job",
    "weekly": "src.scheduler.weekly_job",
    "leads": "src.scheduler.lead_job",
    "engagement": "src.learning.auto_learner",
    "feedback": "src.learning.process_feedback",
}
STATEFUL = ["daily", "weekly", "leads", "onboard", "feedback", "engagement"]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(name: str) -> list[dict]:
    wf = _load(GH / "workflows" / f"{name}.yml")
    return next(iter(wf["jobs"].values()))["steps"]


def _on(wf: dict) -> dict:
    return wf.get("on") or wf.get(True)  # PyYAML parses a bare `on:` key as boolean True


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_every_workflow_parses_and_has_a_trigger(path):
    wf = _load(path)
    assert _on(wf) and wf["jobs"]


def test_every_local_action_and_script_reference_exists():
    root = GH.parent
    for path in [*WORKFLOWS, *(GH / "actions").rglob("action.yml")]:
        text = path.read_text(encoding="utf-8")
        for ref in re.findall(r"uses:\s*\./(\S+)", text):
            assert (root / ref / "action.yml").exists(), f"{path.name}: missing local action {ref}"
        for ref in re.findall(r"bash\s+(\.github/scripts/\S+)", text):
            assert (root / ref).exists(), f"{path.name}: missing script {ref}"


def test_state_encryption_key_is_passed_to_setup_not_checkout():
    for path in WORKFLOWS:
        wf = _load(path)
        steps = [step for job in wf["jobs"].values() for step in job.get("steps", [])]
        for step in steps:
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                assert "state_encryption_key" not in step.get("with", {}), path.name
        setup = next((step for step in steps if step.get("uses") == "./.github/actions/setup-agent"), None)
        has_persist_step = any(step.get("uses") == "./.github/actions/persist-state" for step in steps)
        if setup and has_persist_step:
            assert "state_encryption_key" in setup.get("with", {}), path.name


def test_runner_script_keeps_the_pipeline_exit_status():
    """`python ... | while read` swallowed every SystemExit(1) without pipefail."""
    script = (GH / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "set -o pipefail" in script and "python -m" in script and "| while" in script


@pytest.mark.parametrize("name,module", PIPELINE_WORKFLOWS.items())
def test_pipeline_steps_go_through_the_pipefail_runner(name, module):
    runs = [s.get("run", "") for s in _steps(name)]
    assert any(f"run_pipeline.sh {module}" in r for r in runs), name
    # no hand-rolled `| while` pipes left behind in workflow files
    assert "| while" not in (GH / "workflows" / f"{name}.yml").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", STATEFUL)
def test_stateful_workflows_restore_and_persist_state(name):
    steps = _steps(name)
    uses = [s.get("uses", "") for s in steps]
    assert "./.github/actions/setup-agent" in uses, name
    persist = [s for s in steps if s.get("uses") == "./.github/actions/persist-state"]
    assert len(persist) == 1, name
    assert str(persist[0].get("if")).startswith("always()"), f"{name}: state must be saved even when the run fails"
    assert persist[0]["with"]["files"].strip()


def test_persist_action_stages_files_individually_and_saves_encrypted_cache_always():
    action = _load(GH / "actions" / "persist-state" / "action.yml")
    steps = action["runs"]["steps"]
    commit = next(s for s in steps if s["name"] == "Commit state")["run"]
    # `git add a b` aborts entirely when one path is missing (e.g. no engagement log yet)
    assert 'for f in $FILES' in commit and '[ -e "$f" ]' in commit
    assert "git add data/" not in commit
    assert "git pull --rebase" in commit and "for attempt in 1 2 3" in commit
    save = next(s for s in steps if s.get("uses", "").startswith("actions/cache/save"))
    assert str(save["if"]).startswith("always()")
    assert save["with"]["path"] == ".agent-state.enc"
    assert "agent-db-" not in save["with"]["key"]
    assert any(s.get("name") == "Encrypt private agent state" for s in steps)


def test_setup_action_restores_state_without_saving_early():
    steps = _load(GH / "actions" / "setup-agent" / "action.yml")["runs"]["steps"]
    uses = [s.get("uses", "") for s in steps]
    assert any(u.startswith("actions/cache/restore@") for u in uses) and not any("cache/save" in u for u in uses)
    restore = next(s for s in steps if s.get("uses", "").startswith("actions/cache/restore"))
    assert restore["with"]["path"] == ".agent-state.enc"
    assert any(s.get("name") == "Decrypt private agent state" for s in steps)
    # Reels are best-effort: a missing piper/ffmpeg may warn but must not kill the job
    check = next(s for s in steps if "render dependencies" in s.get("name", ""))
    assert "exit 1" not in check["run"]


@pytest.mark.parametrize("name", ["leads", "onboard"])
def test_dispatch_inputs_are_only_ever_bound_to_env_vars(name):
    """Interpolating ${{ github.event.inputs.* }} into a run script allows shell /
    python injection (and breaks on a quote in a niche like "Dunkin' Donuts")."""
    text = (GH / "workflows" / f"{name}.yml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "${{ github.event.inputs." in line:
            assert re.match(r"^\s+(IN_|INPUT_)\w+: \$\{\{ github\.event\.inputs\.\w+ \}\}\s*$", line) or line.strip().startswith("if:"), line


def test_lead_config_overrides_are_discarded_not_committed():
    persist = next(s for s in _steps("leads") if s.get("uses") == "./.github/actions/persist-state")["with"]
    assert persist["discard"] == "data/lead_config.json"
    assert "lead_config.json" not in persist["files"]


def test_drift_check_can_actually_alert():
    """It used to crash with ModuleNotFoundError behind `| tee` and report success."""
    steps = _steps("drift_check")
    check = next(s for s in steps if s.get("id") == "check")
    assert check["shell"] == "bash" and "| tee" in check["run"]  # `shell: bash` => pipefail
    assert any(s.get("if") == "steps.check.outcome == 'failure'" and "exit 1" in s.get("run", "") for s in steps)
    assert any("gh issue" in s.get("run", "") for s in steps)


def test_feedback_workflow_ignores_unrelated_issues_and_sees_edits():
    wf = _load(GH / "workflows" / "feedback.yml")
    job = wf["jobs"]["feedback"]
    assert "[content-intel-feedback]" in job["if"]
    assert set(_on(wf)["issues"]["types"]) == {"opened", "edited"}
    assert "workflow_dispatch" in _on(wf)


def test_onboarding_is_connected_to_the_daily_run():
    wf = _load(GH / "workflows" / "onboard.yml")
    assert wf["permissions"]["actions"] == "write"
    assert "run_daily_now" in _on(wf)["workflow_dispatch"]["inputs"]
    trigger = next(s for s in _steps("onboard") if "gh workflow run daily.yml" in s.get("run", ""))
    assert "run_daily_now" in trigger["if"]


def test_all_stateful_workflows_share_one_concurrency_group():
    groups = {_load(GH / "workflows" / f"{n}.yml")["concurrency"]["group"] for n in STATEFUL}
    assert groups == {"ai-ads-repo-state"}  # serialised: they all commit the same JSON state


def test_ci_runs_tests_and_lints_the_workflows():
    ci = _load(GH / "workflows" / "ci.yml")
    assert set(ci["jobs"]) == {"test", "workflows", "dependencies", "composite-actions"}
    text = (GH / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest" in text and "ruff" in text and "mypy" in text and "actionlint" in text


def test_secrets_check_is_manual_only():
    assert set(_on(_load(GH / "workflows" / "secrets_check.yml"))) == {"workflow_dispatch"}


SITE_PIPELINES = ["daily", "weekly", "leads", "onboard", "feedback"]


@pytest.mark.parametrize("name", SITE_PIPELINES)
def test_pipelines_restore_and_publish_the_site(name):
    steps = _steps(name)
    setup = next(s for s in steps if s.get("uses") == "./.github/actions/setup-agent")
    assert setup["with"]["site"] == "true", f"{name}: must start from the published site"
    publish = [s for s in steps if s.get("uses") == "./.github/actions/publish-site"]
    assert len(publish) == 1 and str(publish[0].get("if")).startswith("always()"), name
    order = [s.get("uses") for s in steps]
    assert order.index("./.github/actions/publish-site") > order.index("./.github/actions/persist-state"), name


@pytest.mark.parametrize("name", SITE_PIPELINES + ["drift_check"])
def test_llm_workflows_can_use_github_models_and_every_free_provider(name):
    wf = _load(GH / "workflows" / f"{name}.yml")
    assert wf["permissions"]["models"] == "read"
    env = next(iter(wf["jobs"].values()))["env"]
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GITHUB_TOKEN"):
        assert var in env, f"{name}: {var} missing"


@pytest.mark.parametrize("name", ["daily", "leads", "feedback"])
def test_image_workflows_pass_every_image_provider_secret(name):
    env = next(iter(_load(GH / "workflows" / f"{name}.yml")["jobs"].values()))["env"]
    for var in ("HF_TOKEN", "TOGETHER_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "GEMINI_API_KEY"):
        assert var in env, f"{name}: {var} missing"


def test_publish_action_is_a_noop_unless_dirty_and_pushes_a_single_orphan_commit():
    action = _load(GH / "actions" / "publish-site" / "action.yml")
    run = action["runs"]["steps"][0]["run"]
    assert "site/.dirty" in run and "exit 0" in run.split("site/.dirty")[1].split("fi")[0]
    assert "git init -q -b gh-pages" in run and "git push -f origin gh-pages" in run  # no history bloat
    assert "rm -f site/.dirty" in run  # the marker itself is never published


def test_setup_action_restores_gh_pages_only_when_asked():
    steps = _load(GH / "actions" / "setup-agent" / "action.yml")["runs"]["steps"]
    restore = next(s for s in steps if "gh-pages" in s.get("name", ""))
    assert restore["if"] == "inputs.site == 'true'"
    assert "--branch gh-pages" in restore["run"] and "rm -rf site/.git" in restore["run"]
    assert "mkdir -p site" in restore["run"]  # first run: no branch yet


def test_site_checkout_is_ignored_but_site_templates_are_not():
    """`site/` (unanchored) also matched templates/site/ and the templates were never committed."""
    lines = (GH.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/site/" in lines and "site/" not in lines
    assert (GH.parent / "templates" / "site" / "base.html").exists()


def test_ci_actionlint_knows_the_models_permission():
    text = (GH / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "v1.7.12" in text  # older releases reject `permissions: models`


def test_setup_action_provides_ffmpeg_without_apt_and_cannot_block_the_job():
    """`sudo apt-get install ffmpeg` hung CI for 6+ minutes; the same step runs in production
    pipelines, so it must be bounded, non-fatal and independent of apt mirrors."""
    steps = _load(GH / "actions" / "setup-agent" / "action.yml")["runs"]["steps"]
    step = next(s for s in steps if "ffmpeg" in s.get("name", "") and "Check" not in s["name"])
    assert step["if"] == "inputs.piper == 'true'"
    assert "timeout-minutes" not in step  # invalid on composite steps: the runner rejects the action file
    assert "timeout 300" in step["run"] and step["continue-on-error"] is True
    assert "apt-get" not in step["run"] and "imageio-ffmpeg" in step["run"]
    assert "command -v ffmpeg" in step["run"] and "GITHUB_PATH" in step["run"]


def test_no_workflow_or_action_uses_apt_get():
    for path in [*(GH / "workflows").glob("*.yml"), *(GH / "actions").rglob("action.yml")]:
        assert "apt-get" not in path.read_text(encoding="utf-8"), f"{path.name}: apt can stall a run"


def test_ci_ffmpeg_step_is_bounded_and_non_fatal():
    ci = _load(GH / "workflows" / "ci.yml")
    steps = ci["jobs"]["test"]["steps"]
    ff = next(s for s in steps if "ffmpeg" in s.get("name", ""))
    assert ff["timeout-minutes"] <= 5 and ff["continue-on-error"] is True
    assert ci["jobs"]["test"]["timeout-minutes"] <= 15


@pytest.mark.parametrize("name", SITE_PIPELINES)
def test_security_settings_reach_every_pipeline(name):
    """Fail-closed settings are useless if the workflow never passes them (audit C-03/H-01)."""
    env = next(iter(_load(GH / "workflows" / f"{name}.yml")["jobs"].values()))["env"]
    for var in ("FEEDBACK_SIGNING_KEY", "FEEDBACK_ALLOWED_USERS", "OWNER_EMAIL", "DIGEST_EMAIL", "LLM_KEYLESS_FALLBACK", "BELOW_THRESHOLD_POLICY"):
        assert var in env, f"{name}: {var} is not passed to the pipeline"
    assert "secrets." in env["FEEDBACK_SIGNING_KEY"] and "vars." in env["FEEDBACK_ALLOWED_USERS"]


@pytest.mark.parametrize("name", ["daily", "weekly", "leads", "onboard", "feedback", "engagement"])
def test_the_learning_ledger_is_persisted_wherever_weights_are(name):
    persist = next(s for s in _steps(name) if s.get("uses") == "./.github/actions/persist-state")["with"]["files"].split()
    if "data/weights.json" in persist:
        assert "data/learning_ledger.jsonl" in persist, f"{name}: weights are committed without their ledger (no rollback possible)"


def test_hourly_feedback_job_skips_its_expensive_setup_when_nothing_is_pending():
    """Measured: setup-agent costs ~40 s per job; an hourly run with no feedback paid it to do nothing."""
    steps = _steps("feedback")
    names = [s.get("name") or s.get("uses") for s in steps]
    precheck = next(s for s in steps if s.get("id") == "pending")
    assert "gh issue list" in precheck["run"] and "GITHUB_OUTPUT" in precheck["run"]
    assert names.index(precheck["name"]) < names.index("./.github/actions/setup-agent")  # cheap check FIRST
    gate = "steps.pending.outputs.count != '0'"
    for s in steps:
        if s is precheck or str(s.get("uses", "")).startswith("actions/checkout@"):
            continue
        if s.get("name") == "Alert on pipeline failure":
            assert s.get("if") == "failure()"  # a successful empty run has no failure to alert on
            continue
        assert gate in str(s.get("if")), f"step {s.get('name') or s.get('uses')!r} would run even with nothing to process"


def test_ci_gates_on_coverage_and_scans_dependencies():
    text = (GH / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--cov-fail-under=" in text
    assert "pip-audit" in text
    assert (GH / "dependabot.yml").exists()


def test_third_party_actions_are_pinned_to_a_commit_sha():
    """A moving tag (`@v4`) lets whoever controls it change what runs with our token."""
    import re

    bad = []
    for f in list(GH.rglob("*.yml")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.search(r"uses:\s*([^\s#]+)", line)
            if m and not m.group(1).startswith("./") and not re.search(r"@[0-9a-f]{40}$", m.group(1)):
                bad.append((f.name, m.group(1)))
    assert not bad, bad


def test_publish_never_puts_the_token_in_the_remote_url_and_keeps_a_rollback_branch():
    text = (GH / "actions" / "publish-site" / "action.yml").read_text(encoding="utf-8")
    assert "x-access-token:${GH_TOKEN}@" not in text
    assert "extraheader" in text and "add-mask" in text
    assert "gh-pages-backup" in text  # previous site kept as a rollback point
    assert "ls-remote" in text  # post-publish verification


def test_every_state_writing_workflow_shares_one_concurrency_group():
    """persist-state resolves conflicts with `-X theirs`; that is only safe while writers are serialised."""
    writers = [f for f in (GH / "workflows").glob("*.yml") if "persist-state" in f.read_text(encoding="utf-8")]
    assert len(writers) >= 5
    for f in writers:
        assert "group: ai-ads-repo-state" in f.read_text(encoding="utf-8"), f.name


def test_ledger_merges_by_union_so_concurrent_appends_never_conflict():
    attrs = (GH.parent / ".gitattributes").read_text(encoding="utf-8")
    assert "data/learning_ledger.jsonl merge=union" in attrs


# keys a step of a COMPOSITE action may use; anything else makes the runner reject the whole action file
# (this shipped once: `timeout-minutes` on a composite step killed every pipeline in 12 s, and neither
# actionlint nor the YAML parser noticed).
COMPOSITE_STEP_KEYS = {"id", "if", "name", "run", "shell", "uses", "with", "env", "working-directory", "continue-on-error"}


def test_composite_action_steps_use_only_keys_the_runner_accepts():
    for action in sorted((GH / "actions").glob("*/action.yml")):
        for step in _load(action)["runs"]["steps"]:
            extra = set(step) - COMPOSITE_STEP_KEYS
            assert not extra, f"{action.parent.name}: step {step.get('name') or step.get('uses')!r} has unsupported keys {sorted(extra)}"


def test_scheduled_workflows_alert_inline_only_after_a_failure():
    """Successful schedules must not spawn a second, skipped workflow run."""
    assert not (GH / "workflows" / "alert_on_failure.yml").exists()
    for path in (GH / "workflows").glob("*.yml"):
        wf = _load(path)
        triggers = wf.get(True, wf.get("on")) or {}
        if isinstance(triggers, dict) and "schedule" in triggers:
            steps = next(iter(wf["jobs"].values()))["steps"]
            alert_steps = [s for s in steps if s.get("name") == "Alert on pipeline failure"]
            assert len(alert_steps) == 1, f"{path.name} has no inline failure alert"
            assert alert_steps[0].get("if") == "failure()", f"{path.name} alerts only when failed"
            assert "gh issue create" in alert_steps[0].get("run", ""), f"{path.name} alert must work without checkout"
            assert alert_steps[0].get("env", {}).get("GH_TOKEN") == "${{ github.token }}", path
            assert wf["permissions"]["issues"] == "write", f"{path.name} cannot open an alert issue"
