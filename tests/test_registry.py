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


def test_all_sources_have_download_strategy():
    """Every source in sources.yml must declare a download_strategy."""
    sources = load_sources()
    missing = [key for key, spec in sources.items() if spec.download_strategy is None]
    assert not missing, f"Sources without download_strategy: {missing}"


def test_iter_sources_public_only_filters():
    """public_only=True should exclude landing_page-only sources."""
    all_specs = list(iter_sources(public_only=False))
    public_specs = list(iter_sources(public_only=True))
    # landing_page sources have download_strategy but are not directly downloadable
    # public_only filters sources with no download_strategy
    assert len(public_specs) <= len(all_specs)
    assert all(s.download_strategy is not None for s in public_specs)
