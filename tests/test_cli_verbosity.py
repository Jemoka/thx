"""The CLI installs a Loguru stderr sink with predictable verbosity."""

import json

import pytest
from loguru import logger
from typer.testing import CliRunner

from theseus.cli.app import app


@pytest.mark.parametrize("flags", [[], ["-v"], ["-vv"], ["-vvv"], ["--verbose"]])
def test_cli_logging_levels_and_output_streams(monkeypatch, flags):
    monkeypatch.setattr(app, "registered_commands", list(app.registered_commands))

    @app.command("logging-probe")
    def probe():
        logger.debug("debug diagnostic")
        logger.info("info progress")
        logger.warning("warning message")
        print('{"ok": true}')

    result = CliRunner().invoke(app, [*flags, "logging-probe"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}
    assert "info progress" in result.stderr
    assert "warning message" in result.stderr
    assert ("debug diagnostic" in result.stderr) == bool(flags)


def test_cli_logging_resets_between_invocations(monkeypatch):
    monkeypatch.setattr(app, "registered_commands", list(app.registered_commands))

    @app.command("logging-probe")
    def probe():
        logger.debug("debug diagnostic")
        logger.info("info progress")

    runner = CliRunner()
    assert runner.invoke(app, ["-v", "logging-probe"]).exit_code == 0
    result = runner.invoke(app, ["logging-probe"])
    assert result.exit_code == 0, result.output
    assert "debug diagnostic" not in result.stderr
    assert result.stderr.count("info progress") == 1
