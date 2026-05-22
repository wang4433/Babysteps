"""Stage-4 package importability smoke test."""


def test_stage4_package_imports():
    import babysteps.stage4  # noqa: F401


def test_stage4_package_docstring_names_firewall():
    import babysteps.stage4
    doc = (babysteps.stage4.__doc__ or "").lower()
    assert "stage 4" in doc
    assert "privileged" in doc or "firewall" in doc
