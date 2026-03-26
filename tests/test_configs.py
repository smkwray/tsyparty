from tsyparty.config import load_yaml


def test_sources_config_loads():
    payload = load_yaml("configs/sources.yml")
    assert "sources" in payload
    assert "z1_release_page" in payload["sources"]


def test_inference_config_loads():
    payload = load_yaml("configs/inference.yml")
    assert payload["canonical_frequency"] == "quarterly"
