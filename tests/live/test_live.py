import os

import pytest

from cachekeeper.adapters.inbound import cli

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("CACHEKEEPER_LIVE") != "1", reason="needs CACHEKEEPER_LIVE=1"),
]


@pytest.mark.parametrize(
    ("upstream", "model"),
    [
        (os.environ.get("CACHEKEEPER_OLLAMA", "http://localhost:11434"), "qwen3:4b"),
        (os.environ.get("CACHEKEEPER_LLAMACPP", "http://localhost:18080"), "qwen3:4b"),
    ],
)
def test_probe_warm_request_hits(upstream: str, model: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["probe", "--upstream", upstream, "--model", model]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")][1:]
    hit_column = [row.split("|")[4].strip() for row in rows]
    assert hit_column[0].startswith("0%")
    assert int(hit_column[1].split("%")[0]) >= 90
    assert int(hit_column[3].split("%")[0]) < 10
