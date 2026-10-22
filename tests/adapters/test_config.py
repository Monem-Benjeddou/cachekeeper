from pathlib import Path

import pytest
from pydantic import ValidationError

from cachekeeper.adapters.inbound import cli
from cachekeeper.config import Settings, config_file


def write_config(text: str) -> Path:
    path = config_file()
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def test_config_file_lives_under_home(isolated_home: Path) -> None:
    assert config_file() == isolated_home / ".config" / "cachekeeper" / "config.toml"


def test_toml_then_env_then_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    write_config('upstream = "http://toml:1"\nport = 5000\nretention_days = 3\n')
    assert (Settings().upstream, Settings().port, Settings().retention_days) == ("http://toml:1", 5000, 3)
    monkeypatch.setenv("CACHEKEEPER_PORT", "6000")
    monkeypatch.setenv("CACHEKEEPER_RETENTION_DAYS", "")
    assert (Settings().port, Settings().retention_days) == (6000, 3)
    assert Settings(port=7000).port == 7000


def test_invalid_toml_values_fail_at_startup(capsys: pytest.CaptureFixture[str]) -> None:
    write_config('port = "high"\nunknown_key = 1\n')
    assert cli.main(["serve"]) == 2
    errors = capsys.readouterr().err
    assert "port" in errors
    assert "unknown_key" in errors


def test_missing_config_file_uses_defaults() -> None:
    assert Settings().port == 4141
    with pytest.raises(ValidationError):
        Settings(backend="vllm")  # type: ignore[arg-type]
