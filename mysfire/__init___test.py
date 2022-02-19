def test_import() -> None:
    import mysfire  # noqa: F401


def test_version() -> None:
    import mysfire

    assert mysfire.__version__ is not None
