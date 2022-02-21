def test_import() -> None:
    import mysfire  # noqa: F401


def test_version() -> None:
    import mysfire

    assert mysfire.__version__ is not None


def test_processor_imports() -> None:
    from mysfire.processors import PROCESSORS, Processor

    for typestr in ("str", "int", "float", "str.list", "nlp.vocab_tokenization"):
        assert typestr in PROCESSORS
        assert issubclass(PROCESSORS[typestr], Processor)
