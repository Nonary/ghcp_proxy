import json
import os
import shutil
import unittest
import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import auto_update


class AutoUpdateTests(unittest.TestCase):
    _temp_counter = 0

    @contextmanager
    def _tempdir(self):
        # Keep temp writes inside the repository and avoid tempfile's
        # restrictive Windows ACL handling, which can break sandboxed test runs.
        AutoUpdateTests._temp_counter += 1
        path = Path(os.getcwd()) / "__test_auto_update_tmp__" / f"case-{AutoUpdateTests._temp_counter}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass

    def _manager(self, runner, temp_dir, *, now=1000.0):
        root = Path(temp_dir)
        return auto_update.AutoUpdateManager(
            repo_dir=str(root / "repo"),
            state_file=str(root / "state" / "auto-update.json"),
            settings_file=str(root / "state" / "auto-update-settings.json"),
            lock_file=str(root / "state" / "auto-update.lock"),
            command_runner=runner,
            clock=lambda: now,
        )

    def test_disabled_auto_update_skips_without_git(self):
        calls = []

        def runner(command):
            calls.append(command)
            return auto_update.GitCommandResult(0, "")

        with self._tempdir() as temp_dir, mock.patch.dict(os.environ, {"GHCP_AUTO_UPDATE": "0"}):
            manager = self._manager(runner, temp_dir)
            result = manager.startup_check_and_update(force=True)

        self.assertEqual(result["reason"], "disabled")
        self.assertFalse(result["attempted"])
        self.assertEqual(calls, [])

    def test_fast_forwards_when_clean_and_behind_upstream(self):
        calls = []
        head = {"value": "oldsha"}

        def runner(command):
            calls.append(tuple(command[3:]))
            args = tuple(command[3:])
            if args == ("rev-parse", "--is-inside-work-tree"):
                return auto_update.GitCommandResult(0, "true\n")
            if args == ("rev-parse", "--show-toplevel"):
                return auto_update.GitCommandResult(0, "/repo\n")
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return auto_update.GitCommandResult(0, "main\n")
            if args == ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"):
                return auto_update.GitCommandResult(0, "origin/main\n")
            if args == ("fetch", "--quiet", "--prune", "origin"):
                return auto_update.GitCommandResult(0, "")
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return auto_update.GitCommandResult(0, "")
            if args == ("rev-list", "--left-right", "--count", "HEAD...@{u}"):
                return auto_update.GitCommandResult(0, "0 2\n")
            if args == ("rev-parse", "HEAD"):
                return auto_update.GitCommandResult(0, head["value"] + "\n")
            if args == ("merge", "--ff-only", "origin/main"):
                head["value"] = "newsha"
                return auto_update.GitCommandResult(0, "Updating oldsha..newsha\n")
            raise AssertionError(f"unexpected git args: {args}")

        with self._tempdir() as temp_dir, mock.patch.dict(
            os.environ,
            {"GHCP_AUTO_UPDATE": "1", "GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "0"},
        ):
            manager = self._manager(runner, temp_dir)
            result = manager.startup_check_and_update()

            state = json.loads(Path(manager.state_file).read_text(encoding="utf-8"))

        self.assertTrue(result["updated"])
        self.assertTrue(result["restart_required"])
        self.assertEqual(result["reason"], "updated")
        self.assertEqual(result["old_head"], "oldsha")
        self.assertEqual(result["new_head"], "newsha")
        self.assertIn(("merge", "--ff-only", "origin/main"), calls)
        self.assertEqual(state["last_result"]["reason"], "updated")

    def test_local_tracked_changes_skip_before_merge(self):
        calls = []

        def runner(command):
            calls.append(tuple(command[3:]))
            args = tuple(command[3:])
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): auto_update.GitCommandResult(0, "true\n"),
                ("rev-parse", "--show-toplevel"): auto_update.GitCommandResult(0, "/repo\n"),
                ("rev-parse", "--abbrev-ref", "HEAD"): auto_update.GitCommandResult(0, "main\n"),
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): auto_update.GitCommandResult(0, "origin/main\n"),
                ("fetch", "--quiet", "--prune", "origin"): auto_update.GitCommandResult(0, ""),
                ("rev-list", "--left-right", "--count", "HEAD...@{u}"): auto_update.GitCommandResult(0, "0 2\n"),
                ("status", "--porcelain", "--untracked-files=all"): auto_update.GitCommandResult(0, " M proxy.py\n"),
            }
            if args in mapping:
                return mapping[args]
            raise AssertionError(f"unexpected git args: {args}")

        with self._tempdir() as temp_dir, mock.patch.dict(
            os.environ,
            {"GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "0", "GHCP_AUTO_UPDATE_MODE": "developer"},
        ):
            manager = self._manager(runner, temp_dir)
            result = manager.startup_check_and_update()

        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "local-changes")
        self.assertTrue(result["upgrade_blocked"])
        self.assertNotIn(("merge", "--ff-only", "origin/main"), calls)

    def test_user_mode_stashes_local_changes_and_reapplies_after_upgrade(self):
        calls = []
        head = {"value": "oldsha"}

        def runner(command):
            calls.append(tuple(command[3:]))
            args = tuple(command[3:])
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): auto_update.GitCommandResult(0, "true\n"),
                ("rev-parse", "--show-toplevel"): auto_update.GitCommandResult(0, "/repo\n"),
                ("rev-parse", "--abbrev-ref", "HEAD"): auto_update.GitCommandResult(0, "main\n"),
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): auto_update.GitCommandResult(0, "origin/main\n"),
                ("fetch", "--quiet", "--prune", "origin"): auto_update.GitCommandResult(0, ""),
                ("rev-list", "--left-right", "--count", "HEAD...@{u}"): auto_update.GitCommandResult(0, "0 1\n"),
                ("status", "--porcelain", "--untracked-files=all"): auto_update.GitCommandResult(0, " M proxy.py\n?? new_tool.py\n"),
                ("stash", "apply", "stash@{0}"): auto_update.GitCommandResult(0, ""),
                ("stash", "drop", "stash@{0}"): auto_update.GitCommandResult(0, ""),
            }
            if args == ("rev-parse", "HEAD"):
                return auto_update.GitCommandResult(0, head["value"] + "\n")
            if args == ("merge", "--ff-only", "origin/main"):
                head["value"] = "newsha"
                return auto_update.GitCommandResult(0, "Updating oldsha..newsha\n")
            if len(args) >= 7 and args[:4] == ("stash", "push", "--include-untracked", "-m") and args[5:] == ("--", "."):
                return auto_update.GitCommandResult(0, "Saved working directory and index state\n")
            if args in mapping:
                return mapping[args]
            raise AssertionError(f"unexpected git args: {args}")

        with self._tempdir() as temp_dir, mock.patch.dict(
            os.environ,
            {"GHCP_AUTO_UPDATE": "1", "GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "0", "GHCP_AUTO_UPDATE_MODE": "user"},
        ):
            manager = self._manager(runner, temp_dir)
            result = manager.startup_check_and_update()

        self.assertTrue(result["updated"])
        self.assertTrue(result["restart_required"])
        self.assertTrue(result["stashed_local_changes"])
        self.assertIn(("stash", "apply", "stash@{0}"), calls)
        self.assertLess(
            next(i for i, args in enumerate(calls) if args and args[0:2] == ("stash", "push")),
            calls.index(("merge", "--ff-only", "origin/main")),
        )

    def test_override_discards_local_changes_before_upgrade(self):
        calls = []
        head = {"value": "oldsha"}

        def runner(command):
            calls.append(tuple(command[3:]))
            args = tuple(command[3:])
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): auto_update.GitCommandResult(0, "true\n"),
                ("rev-parse", "--show-toplevel"): auto_update.GitCommandResult(0, "/repo\n"),
                ("rev-parse", "--abbrev-ref", "HEAD"): auto_update.GitCommandResult(0, "main\n"),
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): auto_update.GitCommandResult(0, "origin/main\n"),
                ("fetch", "--quiet", "--prune", "origin"): auto_update.GitCommandResult(0, ""),
                ("rev-list", "--left-right", "--count", "HEAD...@{u}"): auto_update.GitCommandResult(0, "0 1\n"),
                ("status", "--porcelain", "--untracked-files=all"): auto_update.GitCommandResult(0, " M proxy.py\n?? new_tool.py\n"),
                ("reset", "--hard", "HEAD"): auto_update.GitCommandResult(0, ""),
                ("clean", "-fd"): auto_update.GitCommandResult(0, ""),
            }
            if args == ("rev-parse", "HEAD"):
                return auto_update.GitCommandResult(0, head["value"] + "\n")
            if args == ("merge", "--ff-only", "origin/main"):
                head["value"] = "newsha"
                return auto_update.GitCommandResult(0, "Updating oldsha..newsha\n")
            if args in mapping:
                return mapping[args]
            raise AssertionError(f"unexpected git args: {args}")

        with self._tempdir() as temp_dir, mock.patch.dict(
            os.environ,
            {"GHCP_AUTO_UPDATE": "1", "GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "0", "GHCP_AUTO_UPDATE_MODE": "user"},
        ):
            manager = self._manager(runner, temp_dir)
            result = manager.startup_check_and_update(override_local_changes=True)

        self.assertTrue(result["updated"])
        self.assertTrue(result["discarded_local_changes"])
        self.assertIn(("reset", "--hard", "HEAD"), calls)
        self.assertIn(("clean", "-fd"), calls)
        self.assertFalse(any(args and args[0:2] == ("stash", "push") for args in calls))

    def test_local_ahead_checkout_is_not_changed(self):
        calls = []

        def runner(command):
            calls.append(tuple(command[3:]))
            args = tuple(command[3:])
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): auto_update.GitCommandResult(0, "true\n"),
                ("rev-parse", "--show-toplevel"): auto_update.GitCommandResult(0, "/repo\n"),
                ("rev-parse", "--abbrev-ref", "HEAD"): auto_update.GitCommandResult(0, "main\n"),
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): auto_update.GitCommandResult(0, "origin/main\n"),
                ("fetch", "--quiet", "--prune", "origin"): auto_update.GitCommandResult(0, ""),
                ("status", "--porcelain", "--untracked-files=all"): auto_update.GitCommandResult(0, ""),
                ("rev-list", "--left-right", "--count", "HEAD...@{u}"): auto_update.GitCommandResult(0, "3 0\n"),
            }
            if args in mapping:
                return mapping[args]
            raise AssertionError(f"unexpected git args: {args}")

        with self._tempdir() as temp_dir, mock.patch.dict(os.environ, {"GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "0"}):
            manager = self._manager(runner, temp_dir)
            result = manager.startup_check_and_update()

        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "local-ahead")
        self.assertEqual(result["ahead"], 3)
        self.assertNotIn(("merge", "--ff-only", "origin/main"), calls)

    def test_recent_state_skips_network_check(self):
        calls = []

        def runner(command):
            calls.append(command)
            return auto_update.GitCommandResult(0, "")

        with self._tempdir() as temp_dir, mock.patch.dict(
            os.environ,
            {"GHCP_AUTO_UPDATE": "1", "GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "3600"},
        ):
            manager = self._manager(runner, temp_dir, now=2000.0)
            Path(manager.state_file).parent.mkdir(parents=True)
            Path(manager.state_file).write_text(
                json.dumps({"last_check_epoch": 1990.0, "last_result": {"reason": "up-to-date"}}),
                encoding="utf-8",
            )
            result = manager.startup_check_and_update()

        self.assertEqual(result["reason"], "recently-checked")
        self.assertEqual(calls, [])

    def test_stale_recent_update_state_runs_check_when_head_moved(self):
        calls = []
        head = {"value": "oldsha"}

        def runner(command):
            calls.append(tuple(command[3:]))
            args = tuple(command[3:])
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): auto_update.GitCommandResult(0, "true\n"),
                ("rev-parse", "--show-toplevel"): auto_update.GitCommandResult(0, "/repo\n"),
                ("rev-parse", "--abbrev-ref", "HEAD"): auto_update.GitCommandResult(0, "main\n"),
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): auto_update.GitCommandResult(0, "origin/main\n"),
                ("fetch", "--quiet", "--prune", "origin"): auto_update.GitCommandResult(0, ""),
                ("status", "--porcelain", "--untracked-files=all"): auto_update.GitCommandResult(0, ""),
                ("rev-list", "--left-right", "--count", "HEAD...@{u}"): auto_update.GitCommandResult(0, "0 1\n"),
            }
            if args == ("rev-parse", "HEAD"):
                return auto_update.GitCommandResult(0, head["value"] + "\n")
            if args == ("merge", "--ff-only", "origin/main"):
                head["value"] = "newsha"
                return auto_update.GitCommandResult(0, "Updating oldsha..newsha\n")
            if args in mapping:
                return mapping[args]
            raise AssertionError(f"unexpected git args: {args}")

        with self._tempdir() as temp_dir, mock.patch.dict(
            os.environ,
            {"GHCP_AUTO_UPDATE": "1", "GHCP_AUTO_UPDATE_INTERVAL_SECONDS": "3600"},
        ):
            manager = self._manager(runner, temp_dir, now=2000.0)
            Path(manager.state_file).parent.mkdir(parents=True)
            Path(manager.state_file).write_text(
                json.dumps(
                    {
                        "last_check_epoch": 1990.0,
                        "last_result": {
                            "reason": "updated",
                            "updated": True,
                            "old_head": "oldsha",
                            "new_head": "newsha",
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = manager.startup_check_and_update()

        self.assertEqual(calls[0], ("rev-parse", "HEAD"))
        self.assertTrue(result["updated"])
        self.assertEqual(result["reason"], "updated")
        self.assertIn(("fetch", "--quiet", "--prune", "origin"), calls)

    def test_default_check_interval_is_fifteen_minutes(self):
        with self._tempdir() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            manager = self._manager(lambda _command: auto_update.GitCommandResult(0, ""), temp_dir)

        self.assertEqual(manager.check_interval_seconds(), 15 * 60)

    def test_runtime_reexec_waits_for_active_requests_to_drain(self):
        class FakeManager:
            def enabled(self):
                return True

            def check_interval_seconds(self):
                return 15 * 60

            def status_payload(self):
                return {"enabled": True}

            def startup_check_and_update(self, **_kwargs):
                return {
                    "attempted": True,
                    "updated": True,
                    "restart_required": True,
                    "reason": "updated",
                }

        async def exercise():
            reexec_calls = []
            controller = auto_update.AutoUpdateRuntimeController(
                FakeManager(),
                reexec_func=lambda: reexec_calls.append("reexec"),
                logger=lambda _message: None,
                restart_delay_seconds=0.001,
            )
            controller.note_request_started("request-1")
            result = await controller.run_due_check()
            self.assertEqual(result["reason"], "updated")
            status = controller.status_payload()["runtime"]
            self.assertTrue(status["restart_pending"])
            self.assertFalse(status["restart_scheduled"])
            self.assertFalse(status["restarting"])
            self.assertEqual(status["active_requests"], 1)
            self.assertEqual(reexec_calls, [])

            controller.note_request_finished("request-1")
            await asyncio.sleep(0.01)

            status = controller.status_payload()["runtime"]
            self.assertFalse(status["restart_pending"])
            self.assertFalse(status["restart_scheduled"])
            self.assertEqual(reexec_calls, ["reexec"])

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
