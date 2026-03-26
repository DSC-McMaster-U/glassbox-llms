"""
Tests for the CLI module — parse-level and dispatch tests that don't
require model downloads.
"""

import pytest
from glassboxllms.cli.main import main


class TestCLIHelp:
    def test_no_args_shows_help(self, capsys):
        """No arguments should print help and exit 0."""
        ret = main([])
        assert ret == 0

    def test_experiments_command(self, capsys):
        """'experiments' should list registered experiments."""
        ret = main(["experiments"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "logit_lens" in captured.out
        assert "probing" in captured.out
        assert "cot_faithfulness" in captured.out

    def test_unknown_command(self, capsys):
        """Unknown command should exit non-zero."""
        with pytest.raises(SystemExit):
            main(["not-a-command"])
