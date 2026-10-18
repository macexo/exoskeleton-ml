import exo_ml


def test_package_imports() -> None:
    """exo_ml package is installed and importable"""
    assert exo_ml.__name__ == "exo_ml"
