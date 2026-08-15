"""Tests for the combined local application launcher."""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class LaunchAppTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[3]
    launcher = repo_root / "scripts" / "launch_app.sh"

    def test_installs_dependencies_and_starts_both_services(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            bin_path = temporary_path / "bin"
            log_path = temporary_path / "commands.log"
            bin_path.mkdir()

            self._write_executable(
                bin_path / "uv",
                """
                command_line="uv"
                for argument in "$@"; do command_line+=" <${argument}>"; done
                printf '%s\n' "${command_line}" >> "${LAUNCH_TEST_LOG}"
                if [[ "${1:-}" == "run" ]]; then
                  trap 'exit 0' TERM INT
                  while true; do sleep 0.1; done
                fi
                """,
            )
            self._write_executable(
                bin_path / "bun",
                """
                command_line="bun"
                for argument in "$@"; do command_line+=" <${argument}>"; done
                printf '%s\n' "${command_line}" >> "${LAUNCH_TEST_LOG}"
                """,
            )

            environment = os.environ.copy()
            environment["PATH"] = f"{bin_path}{os.pathsep}{environment['PATH']}"
            environment["LAUNCH_TEST_LOG"] = str(log_path)

            completed = subprocess.run(
                [str(self.launcher), "--host", "0.0.0.0"],
                cwd=self.repo_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            commands = log_path.read_text().splitlines()
            self.assertIn("uv <sync> <--frozen>", commands)
            self.assertIn("bun <install> <--frozen-lockfile>", commands)
            self.assertIn(
                "uv <run> <uvicorn> "
                "<packages.vector_search.src.document_preparation.api:app> <--reload>",
                commands,
            )
            self.assertIn(
                "bun <run> <dev> <--> <--host> <0.0.0.0>",
                commands,
            )

    @staticmethod
    def _write_executable(path: Path, body: str) -> None:
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + textwrap.dedent(body).lstrip()
        )
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
