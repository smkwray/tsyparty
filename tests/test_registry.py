"""Tests for registry.py — source loading and spec fields."""

from tsyparty.registry import load_sources, iter_sources, SourceSpec


def test_load_sources_returns_dict():
    sources = load_sources()
    assert isinstance(sources, dict)
    assert len(sources) > 0


def test_load_sources_has_key_sources():
    sources = load_sources()
    assert "z1_release_page" in sources
    assert "soma_holdings_page" in sources
    assert "h8_release_page" in sources


def test_source_spec_fields():
    sources = load_sources()
    spec = sources["z1_release_page"]
    assert isinstance(spec, SourceSpec)
    assert spec.key == "z1_release_page"
    assert spec.category == "federal_reserve"
    assert spec.frequency == "quarterly"
    assert spec.landing_url.startswith("https://")


def test_source_spec_download_strategy():
    sources = load_sources()
    z1 = sources["z1_release_page"]
    assert z1.download_strategy == "discover_artifact"
    soma = sources["soma_holdings_page"]
    assert soma.download_strategy == "json_api"
    assert soma.api_url is not None


def test_iter_sources():
    specs = list(iter_sources())
    assert len(specs) > 10
    assert all(isinstance(s, SourceSpec) for s in specs)
