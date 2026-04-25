"""Best-effort git-based self update support for ghcp_proxy.

The updater intentionally only performs fast-forward updates from the
repository's configured upstream branch. It never rebases, and it only discards
local changes when the dashboard's explicit upgrade override asks it to.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import Callable

from constants import TOKEN_DIR

DEFAULT_CHECK_INTERVAL_SECONDS = 10 * 60
DEFAULT_GIT_TIMEOUT_SECONDS = 60
DEFAULT_NOTICE_INTERVAL_SECONDS = 10 * 60
AUTO_UPDATE_STATE_FILE = os.path.join(TOKEN_DIR, "auto-update.json")
AUTO_UPDATE_SETTINGS_FILE = os.path.join(TOKEN_DIR, "auto-update-settings.json")
AUTO_UPDATE_LOCK_FILE = os.path.join(TOKEN_DIR, "auto-update.lock")
AUTO_UPDATE_MODES = {"user", "developer"}


@dataclass(frozen=True)
class GitCommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


GitCommandRunner = Callable[[list[str]], GitCommandResult]


def _env_flag_default(name: str, *, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


def _env_float(name: str, *, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _single_line(value: str) -> str:
    return str(value or "").strip().splitlines()[0].strip() if str(value or "").strip() else ""


class _UpdateLock:
    def __init__(self, path: str, stale_seconds: float = 10 * 60):
        self.path = path
        self.stale_seconds = stale_seconds
        self._fd: int | None = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(self.path, flags)
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(self.path)
            except OSError:
                age = 0
            if age > self.stale_seconds:
                try:
                    os.remove(self.path)
                except OSError:
                    pass
                try:
                    self._fd = os.open(self.path, flags)
                except FileExistsError:
                    raise RuntimeError("another auto-update check is already running")
            else:
                raise RuntimeError("another auto-update check is already running")

        os.write(self._fd, f"{os.getpid()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            os.remove(self.path)
        except OSError:
            pass


class AutoUpdateManager:
    def __init__(
        self,
        repo_dir: str | None = None,
        *,
        state_file: str = AUTO_UPDATE_STATE_FILE,
        settings_file: str = AUTO_UPDATE_SETTINGS_FILE,
        lock_file: str = AUTO_UPDATE_LOCK_FILE,
        command_runner: GitCommandRunner | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.repo_dir = os.path.abspath(repo_dir or os.path.dirname(__file__))
        self.state_file = state_file
        self.settings_file = settings_file
        self.lock_file = lock_file
        self.command_runner = command_runner or self._run_git
        self.clock = clock

    def enabled(self) -> bool:
        # Default on for normal users, but easy to opt out for packaged,
        # air-gapped, or heavily customized installs.
        return _env_flag_default("GHCP_AUTO_UPDATE", default=True)

    def check_interval_seconds(self) -> float:
        return max(0.0, _env_float("GHCP_AUTO_UPDATE_INTERVAL_SECONDS", default=DEFAULT_CHECK_INTERVAL_SECONDS))

    def mode(self) -> str:
        raw_env = str(os.environ.get("GHCP_AUTO_UPDATE_MODE", "")).strip().lower()
        if raw_env in AUTO_UPDATE_MODES:
            return raw_env
        settings = self._load_settings()
        raw_mode = str(settings.get("mode", "")).strip().lower()
        if raw_mode in AUTO_UPDATE_MODES:
            return raw_mode
        return "user"

    def mode_source(self) -> str:
        raw_env = str(os.environ.get("GHCP_AUTO_UPDATE_MODE", "")).strip().lower()
        if raw_env in AUTO_UPDATE_MODES:
            return "env"
        settings = self._load_settings()
        raw_mode = str(settings.get("mode", "")).strip().lower()
        if raw_mode in AUTO_UPDATE_MODES:
            return "settings"
        return "default"

    def set_mode(self, mode: str) -> dict[str, object]:
        normalized = str(mode or "").strip().lower()
        if normalized not in AUTO_UPDATE_MODES:
            raise ValueError("auto-update mode must be 'user' or 'developer'")
        settings = self._load_settings()
        settings["mode"] = normalized
        self._save_settings(settings)
        return self.settings_payload()

    def settings_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode(),
            "mode_source": self.mode_source(),
            "settings_file": self.settings_file,
            "developer_mode": self.mode() == "developer",
        }

    def status_payload(self) -> dict[str, object]:
        state = self._load_state()
        return {
            "enabled": self.enabled(),
            "repo_dir": self.repo_dir,
            "state_file": self.state_file,
            "settings_file": self.settings_file,
            "check_interval_seconds": self.check_interval_seconds(),
            **self.settings_payload(),
            "last_check": state.get("last_check"),
            "last_result": state.get("last_result"),
        }

    def startup_check_for_update(self, *, force: bool = False) -> dict[str, object]:
        if not self.enabled():
            return {
                "attempted": False,
                "updated": False,
                "update_available": False,
                "restart_required": False,
                "reason": "disabled",
            }

        state = self._load_state()
        now = self.clock()
        if (
            not force
            and not self._state_result_requires_restart(state)
            and not self._is_check_due(state, now)
            and self._recent_state_matches_checkout(state)
        ):
            last_result = state.get("last_result") if isinstance(state.get("last_result"), dict) else {}
            return {
                "attempted": False,
                "updated": False,
                "update_available": bool(last_result.get("update_available")),
                "restart_required": False,
                "reason": "recently-checked",
                "last_check": state.get("last_check"),
                "last_result": state.get("last_result"),
            }

        try:
            with _UpdateLock(self.lock_file):
                result = self.check_for_update()
        except (OSError, RuntimeError) as exc:
            result = {
                "attempted": False,
                "updated": False,
                "update_available": False,
                "restart_required": False,
                "reason": "locked" if isinstance(exc, RuntimeError) else "lock-failed",
                "error": str(exc),
            }

        self._save_state(result, now=self.clock())
        return result

    def startup_check_and_update(
        self,
        *,
        force: bool = False,
        override_local_changes: bool = False,
    ) -> dict[str, object]:
        if not self.enabled():
            return {
                "attempted": False,
                "updated": False,
                "restart_required": False,
                "reason": "disabled",
            }

        state = self._load_state()
        now = self.clock()
        if (
            not force
            and not self._state_result_requires_restart(state)
            and not self._is_check_due(state, now)
            and self._recent_state_matches_checkout(state)
        ):
            return {
                "attempted": False,
                "updated": False,
                "restart_required": False,
                "reason": "recently-checked",
                "last_check": state.get("last_check"),
                "last_result": state.get("last_result"),
            }

        try:
            with _UpdateLock(self.lock_file):
                result = self.check_and_update(override_local_changes=override_local_changes)
        except (OSError, RuntimeError) as exc:
            result = {
                "attempted": False,
                "updated": False,
                "restart_required": False,
                "reason": "locked" if isinstance(exc, RuntimeError) else "lock-failed",
                "error": str(exc),
            }

        self._save_state(result, now=self.clock())
        return result

    def check_for_update(self) -> dict[str, object]:
        repo_check = self._git("rev-parse", "--is-inside-work-tree")
        if repo_check.returncode != 0 or _single_line(repo_check.stdout).lower() != "true":
            return self._result(False, "not-a-git-repo", update_available=False, error=repo_check.stderr)

        toplevel = self._git("rev-parse", "--show-toplevel")
        repo_root = _single_line(toplevel.stdout) if toplevel.returncode == 0 else self.repo_dir

        branch_result = self._git("rev-parse", "--abbrev-ref", "HEAD")
        branch = _single_line(branch_result.stdout) if branch_result.returncode == 0 else ""
        if branch == "HEAD":
            return self._result(False, "detached-head", update_available=False, branch=branch)

        upstream_result = self._git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream_result.returncode != 0:
            return self._result(False, "no-upstream", update_available=False, branch=branch, error=upstream_result.stderr)
        upstream = _single_line(upstream_result.stdout)
        remote = self._remote_from_upstream(upstream)
        if not remote:
            return self._result(False, "no-upstream-remote", update_available=False, branch=branch, upstream=upstream)

        fetch_result = self._git("fetch", "--quiet", "--prune", remote)
        if fetch_result.returncode != 0:
            return self._result(
                False,
                "fetch-failed",
                update_available=False,
                branch=branch,
                upstream=upstream,
                remote=remote,
                error=fetch_result.stderr or fetch_result.stdout,
            )

        counts = self._ahead_behind()
        if counts is None:
            return self._result(False, "ahead-behind-failed", update_available=False, branch=branch, upstream=upstream, remote=remote)
        ahead, behind = counts
        dirty = self._worktree_status(include_untracked=True)
        base_payload = {
            "branch": branch,
            "upstream": upstream,
            "remote": remote,
            "ahead": ahead,
            "behind": behind,
            "repo_root": repo_root,
            "mode": self.mode(),
            "dirty": dirty,
            "update_available": behind > 0,
        }
        if behind > 0:
            return self._result(False, "update-available", **base_payload)
        reason = "local-ahead" if ahead > 0 else "up-to-date"
        return self._result(False, reason, **base_payload)

    def check_and_update(self, *, override_local_changes: bool = False) -> dict[str, object]:
        repo_check = self._git("rev-parse", "--is-inside-work-tree")
        if repo_check.returncode != 0 or _single_line(repo_check.stdout).lower() != "true":
            return self._result(False, "not-a-git-repo", error=repo_check.stderr)

        toplevel = self._git("rev-parse", "--show-toplevel")
        repo_root = _single_line(toplevel.stdout) if toplevel.returncode == 0 else self.repo_dir

        branch_result = self._git("rev-parse", "--abbrev-ref", "HEAD")
        branch = _single_line(branch_result.stdout) if branch_result.returncode == 0 else ""
        if branch == "HEAD":
            return self._result(False, "detached-head", branch=branch)

        upstream_result = self._git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream_result.returncode != 0:
            return self._result(False, "no-upstream", branch=branch, error=upstream_result.stderr)
        upstream = _single_line(upstream_result.stdout)
        remote = self._remote_from_upstream(upstream)
        if not remote:
            return self._result(False, "no-upstream-remote", branch=branch, upstream=upstream)

        fetch_result = self._git("fetch", "--quiet", "--prune", remote)
        if fetch_result.returncode != 0:
            return self._result(
                False,
                "fetch-failed",
                branch=branch,
                upstream=upstream,
                remote=remote,
                error=fetch_result.stderr or fetch_result.stdout,
            )

        counts = self._ahead_behind()
        if counts is None:
            return self._result(False, "ahead-behind-failed", branch=branch, upstream=upstream, remote=remote)
        ahead, behind = counts
        mode = self.mode()
        dirty = self._worktree_status(include_untracked=True)
        base_payload = {
            "branch": branch,
            "upstream": upstream,
            "remote": remote,
            "ahead": ahead,
            "behind": behind,
            "repo_root": repo_root,
            "mode": mode,
        }
        if behind <= 0:
            reason = "local-ahead" if ahead > 0 else "up-to-date"
            return self._result(False, reason, **base_payload, dirty=dirty)

        stash_ref = ""
        stashed = False
        discarded_local_changes = False
        if dirty:
            if "<status-failed>" in dirty:
                return self._blocked_by_local_changes("status-failed", base_payload, dirty)
            if override_local_changes:
                discard_result = self._discard_local_changes()
                if discard_result.returncode != 0:
                    return self._blocked_by_local_changes(
                        "local-changes",
                        base_payload,
                        dirty,
                        error=discard_result.stderr or discard_result.stdout,
                    )
                discarded_local_changes = True
            elif mode == "developer":
                return self._blocked_by_local_changes("local-changes", base_payload, dirty)
            else:
                scoped = self._changes_within_user_scope(dirty, repo_root)
                if not scoped:
                    return self._blocked_by_local_changes("local-changes-outside-user-scope", base_payload, dirty)
                stash_result = self._stash_local_changes()
                if stash_result.returncode != 0:
                    return self._blocked_by_local_changes(
                        "stash-failed",
                        base_payload,
                        dirty,
                        error=stash_result.stderr or stash_result.stdout,
                    )
                stashed = True
                stash_ref = "stash@{0}"

        old_head_result = self._git("rev-parse", "HEAD")
        old_head = _single_line(old_head_result.stdout) if old_head_result.returncode == 0 else ""

        update_method = "rebase" if ahead > 0 else "fast-forward"
        update_result = self._git("rebase", upstream) if ahead > 0 else self._git("merge", "--ff-only", upstream)
        if update_result.returncode != 0:
            abort_result = GitCommandResult(0, "", "")
            if update_method == "rebase":
                abort_result = self._git("rebase", "--abort")
            if stashed:
                self._restore_stash_after_failed_upgrade(stash_ref)
            return self._result(
                False,
                "rebase-failed" if update_method == "rebase" else "fast-forward-failed",
                **base_payload,
                dirty=dirty,
                stashed_local_changes=stashed,
                update_method=update_method,
                error=update_result.stderr or update_result.stdout,
                abort_error=abort_result.stderr or abort_result.stdout,
            )

        new_head_result = self._git("rev-parse", "HEAD")
        new_head = _single_line(new_head_result.stdout) if new_head_result.returncode == 0 else ""

        if stashed:
            apply_result = self._apply_stash(stash_ref)
            if apply_result.returncode != 0:
                rollback_result = self._git("reset", "--hard", old_head) if old_head else GitCommandResult(1, "", "missing old HEAD")
                restore_result = self._apply_stash(stash_ref) if rollback_result.returncode == 0 else GitCommandResult(1, "", "rollback failed")
                if restore_result.returncode == 0:
                    self._drop_stash(stash_ref)
                return self._blocked_by_local_changes(
                    "local-changes-need-commit",
                    base_payload,
                    dirty,
                    error=apply_result.stderr or apply_result.stdout,
                    stash_restored=restore_result.returncode == 0,
                    rollback_error=rollback_result.stderr or rollback_result.stdout,
                    restore_error=restore_result.stderr or restore_result.stdout,
                )
            self._drop_stash(stash_ref)

        return self._result(
            True,
            "updated",
            **base_payload,
            old_head=old_head,
            new_head=new_head,
            dirty=dirty,
            stashed_local_changes=stashed,
            discarded_local_changes=discarded_local_changes,
            update_method=update_method,
            rebased_local_commits=ahead if update_method == "rebase" else 0,
            restart_required=bool(old_head and new_head and old_head != new_head),
        )

    def _result(self, updated: bool, reason: str, **extra: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "attempted": True,
            "updated": updated,
            "restart_required": bool(extra.pop("restart_required", False)),
            "reason": reason,
        }
        payload.update({key: value for key, value in extra.items() if value not in (None, "")})
        return payload

    def _blocked_by_local_changes(
        self,
        reason: str,
        base_payload: dict[str, object],
        dirty: list[str],
        **extra: object,
    ) -> dict[str, object]:
        message = (
            "Auto-update cannot upgrade while local pending changes are present. "
            "Commit or remove the pending changes, switch to user mode so GHCP Proxy can stash and re-apply them, "
            "or use Apply upgrade anyway to discard those pending changes and update."
        )
        if reason == "local-changes-need-commit":
            message = (
                "Auto-update tried to preserve your local changes, but they did not apply cleanly on top of the "
                "upgrade. Your checkout was restored. Commit or remove the pending changes, then try again; "
                "or use Apply upgrade anyway to discard them and update."
            )
        elif reason == "local-changes-outside-user-scope":
            message = (
                "Auto-update found local pending changes outside the user-editable GHCP Proxy folder. "
                "Commit or remove those changes before upgrading, or use Apply upgrade anyway to discard them."
            )
        elif reason == "stash-failed":
            message = (
                "Auto-update could not safely stash local pending changes. Commit or remove them before upgrading, "
                "or use Apply upgrade anyway to discard them."
            )
        elif reason == "status-failed":
            message = (
                "Auto-update could not inspect local pending changes safely. Commit or remove pending changes, "
                "then try again; or use Apply upgrade anyway to discard them and update."
            )
        return self._result(
            False,
            reason,
            **base_payload,
            dirty=dirty,
            upgrade_blocked=True,
            user_message=message,
            **extra,
        )

    def _tracked_worktree_dirty(self) -> list[str]:
        result = self._git("status", "--porcelain", "--untracked-files=no")
        if result.returncode != 0:
            return ["<status-failed>"]
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _worktree_status(self, *, include_untracked: bool) -> list[str]:
        flag = "--untracked-files=all" if include_untracked else "--untracked-files=no"
        result = self._git("status", "--porcelain", flag)
        if result.returncode != 0:
            return ["<status-failed>"]
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _user_change_pathspecs(self, repo_root: str) -> list[str]:
        raw = str(os.environ.get("GHCP_AUTO_UPDATE_USER_CHANGE_PATHS", "")).strip()
        if raw:
            values = [part.strip().replace("\\", "/").strip("/") for part in raw.split(os.pathsep)]
            return [part for part in values if part] or ["."]
        root_name = os.path.basename(os.path.abspath(repo_root or self.repo_dir)).lower()
        if root_name == "ghcp_proxy":
            return ["."]
        if os.path.isdir(os.path.join(repo_root or self.repo_dir, "ghcp_proxy")):
            return ["ghcp_proxy"]
        return ["."]

    def _changes_within_user_scope(self, dirty: list[str], repo_root: str) -> bool:
        pathspecs = self._user_change_pathspecs(repo_root)
        if "." in pathspecs:
            return True
        normalized_specs = [spec.rstrip("/") + "/" for spec in pathspecs]
        for line in dirty:
            for path in self._status_paths(line):
                normalized = path.replace("\\", "/").lstrip("/")
                if not any(normalized == spec.rstrip("/") or normalized.startswith(spec) for spec in normalized_specs):
                    return False
        return True

    def _status_paths(self, line: str) -> list[str]:
        body = str(line or "")[3:].strip()
        if " -> " in body:
            return [part.strip().strip('"') for part in body.split(" -> ", 1) if part.strip()]
        return [body.strip('"')] if body else []

    def _stash_local_changes(self) -> GitCommandResult:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock()))
        return self._git(
            "stash",
            "push",
            "--include-untracked",
            "-m",
            f"ghcp_proxy auto-update {stamp}",
            "--",
            *self._user_change_pathspecs(self.repo_dir),
        )

    def _apply_stash(self, stash_ref: str) -> GitCommandResult:
        return self._git("stash", "apply", stash_ref)

    def _drop_stash(self, stash_ref: str) -> GitCommandResult:
        return self._git("stash", "drop", stash_ref)

    def _restore_stash_after_failed_upgrade(self, stash_ref: str) -> None:
        restore_result = self._apply_stash(stash_ref)
        if restore_result.returncode == 0:
            self._drop_stash(stash_ref)

    def _discard_local_changes(self) -> GitCommandResult:
        reset_result = self._git("reset", "--hard", "HEAD")
        if reset_result.returncode != 0:
            return reset_result
        return self._git("clean", "-fd")

    def _ahead_behind(self) -> tuple[int, int] | None:
        result = self._git("rev-list", "--left-right", "--count", "HEAD...@{u}")
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split()
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    def _remote_from_upstream(self, upstream: str) -> str:
        upstream = upstream.strip()
        if upstream.startswith("refs/remotes/"):
            upstream = upstream[len("refs/remotes/") :]
        if "/" not in upstream:
            return ""
        return upstream.split("/", 1)[0]

    def _git(self, *args: str) -> GitCommandResult:
        return self.command_runner(["git", "-C", self.repo_dir, *args])

    def _run_git(self, command: list[str]) -> GitCommandResult:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_ASKPASS", "")
        env.setdefault("SSH_ASKPASS", "")
        timeout = max(5.0, _env_float("GHCP_AUTO_UPDATE_GIT_TIMEOUT_SECONDS", default=DEFAULT_GIT_TIMEOUT_SECONDS))
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            return GitCommandResult(127, "", str(exc))
        except subprocess.TimeoutExpired as exc:
            return GitCommandResult(124, exc.stdout or "", exc.stderr or f"git command timed out after {timeout:g}s")
        except OSError as exc:
            return GitCommandResult(1, "", str(exc))
        return GitCommandResult(completed.returncode, completed.stdout or "", completed.stderr or "")

    def _is_check_due(self, state: dict[str, object], now: float) -> bool:
        interval = self.check_interval_seconds()
        if interval <= 0:
            return True
        last_check = state.get("last_check_epoch")
        if not isinstance(last_check, (int, float)):
            return True
        return now - float(last_check) >= interval

    def _recent_state_matches_checkout(self, state: dict[str, object]) -> bool:
        expected_head = self._state_expected_head(state)
        if not expected_head:
            return True
        current_head_result = self._git("rev-parse", "HEAD")
        if current_head_result.returncode != 0:
            return False
        return _single_line(current_head_result.stdout) == expected_head

    def _state_expected_head(self, state: dict[str, object]) -> str:
        result = state.get("last_result")
        if not isinstance(result, dict):
            return ""
        for key in ("head", "new_head", "old_head"):
            value = _single_line(str(result.get(key) or ""))
            if value:
                return value
        return ""

    def _state_result_requires_restart(self, state: dict[str, object]) -> bool:
        result = state.get("last_result")
        if not isinstance(result, dict):
            return False
        if bool(result.get("restart_required")):
            return True
        return str(result.get("reason") or "") == "updated"

    def _load_state(self) -> dict[str, object]:
        try:
            with open(self.state_file, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_settings(self) -> dict[str, object]:
        try:
            with open(self.settings_file, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_settings(self, settings: dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        temp_path = f"{self.settings_file}.tmp"
        with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(settings, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_path, self.settings_file)

    def _save_state(self, result: dict[str, object], *, now: float) -> None:
        payload = {
            "last_check_epoch": now,
            "last_check": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "last_result": result,
        }
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            temp_path = f"{self.state_file}.tmp"
            with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(temp_path, self.state_file)
        except OSError:
            pass


class AutoUpdateRuntimeController:
    """Coordinates periodic update checks and deferred restarts at runtime."""

    def __init__(
        self,
        manager: AutoUpdateManager,
        *,
        reexec_func: Callable[[], None] | None = None,
        logger: Callable[[str], None] | None = None,
        restart_delay_seconds: float = 0.25,
        notice_interval_seconds: float = DEFAULT_NOTICE_INTERVAL_SECONDS,
    ):
        self.manager = manager
        self.reexec_func = reexec_func or reexec_current_process
        self.logger = logger or (lambda message: print(message, flush=True))
        self.restart_delay_seconds = max(0.0, float(restart_delay_seconds))
        self.notice_interval_seconds = max(0.0, float(notice_interval_seconds))
        self._lock = RLock()
        self._active_request_ids: set[str] = set()
        self._task: asyncio.Task | None = None
        self._restart_handle: asyncio.Handle | None = None
        self._check_in_progress = False
        self._pending_restart = False
        self._restarting = False
        self._last_runtime_result: dict[str, object] | None = None
        self._last_restart_error: str | None = None
        self._last_notice_epoch = 0.0

    def status_payload(self) -> dict[str, object]:
        payload = self.manager.status_payload()
        with self._lock:
            task_running = bool(self._task and not self._task.done())
            update_available = self._update_available_from_result(self._last_runtime_result)
            runtime = {
                "periodic_task_running": task_running,
                "check_in_progress": self._check_in_progress,
                "active_requests": len(self._active_request_ids),
                "update_available": update_available,
                "restart_pending": self._pending_restart,
                "restart_scheduled": bool(self._restart_handle and not self._restart_handle.cancelled()),
                "restarting": self._restarting,
                "last_runtime_result": self._last_runtime_result,
                "last_restart_error": self._last_restart_error,
            }
        payload["runtime"] = runtime
        return payload

    def update_notice_text_if_due(self) -> str:
        with self._lock:
            if self._update_available_from_result(self._last_runtime_result):
                update_available = True
            else:
                manager_status = self.manager.status_payload()
                update_available = self._update_available_from_result(
                    manager_status.get("last_result") if isinstance(manager_status, dict) else None
                )
            if not update_available:
                return ""
            now = float(self.manager.clock())
            if self._last_notice_epoch and now - self._last_notice_epoch < self.notice_interval_seconds:
                return ""
            self._last_notice_epoch = now
        return "By the way, an update is available for GHCP Proxy. Visit http://localhost:8000 to update the proxy from the dashboard."

    def note_request_started(self, request_id: str) -> None:
        request_id = str(request_id or "").strip()
        if not request_id:
            return
        with self._lock:
            self._active_request_ids.add(request_id)

    def note_request_finished(self, request_id: str) -> None:
        request_id = str(request_id or "").strip()
        if not request_id:
            return
        should_try_restart = False
        with self._lock:
            if request_id in self._active_request_ids:
                self._active_request_ids.remove(request_id)
            should_try_restart = self._pending_restart and not self._active_request_ids
        if should_try_restart:
            self._schedule_restart_if_idle("request-drained")

    def start_periodic_checks(self) -> bool:
        if not self.manager.enabled():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        with self._lock:
            if self._task and not self._task.done():
                return False
            self._task = loop.create_task(self._periodic_loop(), name="ghcp-auto-update")
            return True

    async def stop_periodic_checks(self) -> None:
        with self._lock:
            task = self._task
            self._task = None
            restart_handle = self._restart_handle
            self._restart_handle = None
        if restart_handle and not restart_handle.cancelled():
            restart_handle.cancel()
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def run_due_check(
        self,
        *,
        force: bool = False,
        override_local_changes: bool = False,
    ) -> dict[str, object]:
        if not self.manager.enabled():
            result = {
                "attempted": False,
                "updated": False,
                "update_available": False,
                "restart_required": False,
                "reason": "disabled",
            }
            with self._lock:
                self._last_runtime_result = result
            return result

        with self._lock:
            if self._check_in_progress:
                return {
                    "attempted": False,
                    "updated": False,
                    "restart_required": False,
                    "reason": "already-running",
                }
            self._check_in_progress = True

        try:
            result = await asyncio.to_thread(
                self.manager.startup_check_for_update,
                force=force,
            )
        finally:
            with self._lock:
                self._check_in_progress = False

        if result.get("attempted"):
            self.logger(f"Auto-update check: {json.dumps(result, default=str)}")

        with self._lock:
            self._last_runtime_result = result
        return result

    async def apply_update(
        self,
        *,
        override_local_changes: bool = False,
    ) -> dict[str, object]:
        if not self.manager.enabled():
            result = {
                "attempted": False,
                "updated": False,
                "update_available": False,
                "restart_required": False,
                "reason": "disabled",
            }
            with self._lock:
                self._last_runtime_result = result
            return result

        with self._lock:
            if self._check_in_progress:
                return {
                    "attempted": False,
                    "updated": False,
                    "restart_required": False,
                    "reason": "already-running",
                }
            self._check_in_progress = True

        try:
            result = await asyncio.to_thread(
                self.manager.startup_check_and_update,
                force=True,
                override_local_changes=override_local_changes,
            )
        finally:
            with self._lock:
                self._check_in_progress = False

        if result.get("attempted"):
            self.logger(f"Auto-update apply: {json.dumps(result, default=str)}")

        with self._lock:
            self._last_runtime_result = result
            if result.get("restart_required"):
                self._pending_restart = True
                self._last_restart_error = None
        return result

    def restart_when_idle(self, reason: str = "dashboard") -> bool:
        with self._lock:
            self._pending_restart = True
            self._last_restart_error = None
        return self._schedule_restart_if_idle(reason)

    async def _periodic_loop(self) -> None:
        while True:
            interval = max(1.0, self.manager.check_interval_seconds())
            await asyncio.sleep(interval)
            try:
                await self.run_due_check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive background task guard
                self.logger(f"Warning: auto-update periodic check failed: {exc}")

    def _schedule_restart_if_idle(self, reason: str) -> bool:
        with self._lock:
            if not self._pending_restart or self._restarting or self._active_request_ids:
                return False
            if self._restart_handle and not self._restart_handle.cancelled():
                return False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._restart_if_idle(reason)

        def _restart_callback() -> None:
            with self._lock:
                self._restart_handle = None
            self._restart_if_idle(reason)

        handle = loop.call_later(self.restart_delay_seconds, _restart_callback)
        with self._lock:
            if not self._pending_restart or self._restarting or self._active_request_ids:
                handle.cancel()
                return False
            self._restart_handle = handle
        return True

    def _restart_if_idle(self, reason: str) -> bool:
        with self._lock:
            if not self._pending_restart or self._restarting or self._active_request_ids:
                return False
            self._restarting = True

        self.logger(f"Auto-update applied; restarting GHCP proxy ({reason}).")
        try:
            self.reexec_func()
        except OSError as exc:
            with self._lock:
                self._restarting = False
                self._pending_restart = False
                self._last_restart_error = str(exc)
            self.logger(f"Warning: auto-update restart failed; continuing with current process: {exc}")
            return False

        # os.execv() does not return on success.  Test doubles may return, so
        # clear the pending flag to avoid repeated invocations.
        with self._lock:
            self._restarting = False
            self._pending_restart = False
        return True

    def _update_available_from_result(self, result: dict[str, object] | None) -> bool:
        if not isinstance(result, dict):
            return False
        if bool(result.get("update_available")):
            return True
        return str(result.get("reason") or "") == "update-available"


def run_startup_auto_update(
    repo_dir: str | None = None,
    *,
    force: bool = False,
    override_local_changes: bool = False,
) -> dict[str, object]:
    return AutoUpdateManager(repo_dir=repo_dir).startup_check_and_update(
        force=force,
        override_local_changes=override_local_changes,
    )


def reexec_current_process() -> None:
    os.execv(sys.executable, [sys.executable, *sys.argv])
