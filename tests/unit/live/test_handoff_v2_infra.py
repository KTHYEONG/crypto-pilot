# ruff: noqa
def test_docker_compose_single_daemon() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    text = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "mhs-signal" not in text
    assert "mhs-live" in text
    assert "liquidation-collector" in text
