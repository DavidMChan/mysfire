# flake8: noqa
# type: ignore
import os

from sly import Lexer, Parser


class MysfireHeaderLexer(Lexer):
    tokens = {
        NUMBER,
        FLOAT_NUMBER,
        LPAREN,
        RPAREN,
        COLON,
        COMMA,
        EQUALS,
        QSTRING,
        IDENTIFIER,
        NEWLINE,
        TAB,
        ENV_VAR,
    }
    ignore = " "

    QSTRING = r'"(?:[^"\\]|\\.)*"'
    IDENTIFIER = r"[a-zA-Z_][a-zA-Z0-9_\.]*"
    FLOAT_NUMBER = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    ENV_VAR = r"\$[a-zA-Z_][a-zA-Z0-9_\.]*"
    NUMBER = r"\d+"
    LPAREN = r"\("
    RPAREN = r"\)"
    # LQUOTE = r"\""
    # RQUOTE = r"\""
    COLON = r":"
    COMMA = r","
    EQUALS = r"="
    NEWLINE = r"\n"
    TAB = r"\t"

    def FLOAT_NUMBER(self, t):
        t.value = float(t.value)
        return t

    def NUMBER(self, t):
        t.value = int(t.value)
        return t

    def QSTRING(self, t):
        t.value = t.value[1:-1].replace('\\"', '"')
        return t

    def ENV_VAR(self, t):
        t.value = os.environ.get(t.value[1:], None)
        return t


class MysfireHeaderParser(Parser):
    tokens = MysfireHeaderLexer.tokens

    @_("columns NEWLINE")
    def header(self, p):
        return p.columns

    @_("columns")
    def header(self, p):
        return p.columns

    @_("column")
    def columns(self, p):
        return (p.column,)

    @_("column TAB columns")
    def columns(self, p):
        return (p.column,) + p.columns

    @_("IDENTIFIER COLON IDENTIFIER LPAREN args RPAREN")
    def column(self, p):
        return (p.IDENTIFIER0, p.IDENTIFIER1, p.args)

    @_("IDENTIFIER COLON IDENTIFIER")
    def column(self, p):
        return (p.IDENTIFIER0, p.IDENTIFIER1, None)

    @_("")
    def args(self, p):
        return {}

    @_("argument")
    def args(self, p):
        return p.argument

    @_("argument COMMA args")
    def args(self, p):
        return dict(list(p.argument.items()) + list(p.args.items()))

    @_("IDENTIFIER EQUALS argval")
    def argument(self, p):
        return {p.IDENTIFIER: p.argval}

    # Base argument values
    @_("FLOAT_NUMBER")
    def argval(self, p):
        return p.FLOAT_NUMBER

    @_("NUMBER")
    def argval(self, p):
        return p.VALUE

    @_("QSTRING")
    def argval(self, p):
        return p.QSTRING

    @_("ENV_VAR")
    def argval(self, p):
        return p.ENV_VAR

    def error(self, p):
        if p:
            raise SyntaxError(f"Header syntax error: unexpected '{p.value}' at line {p.lineno}, index {p.index}")
        raise SyntaxError("Header syntax error: unexpected end of file")


if __name__ == "__main__":
    _PARSER_TEST_DATA = r"""
x:str	video:npy(s3_access_key="XXX",s3_secret_key="XXX",s3_endpoint=$ENDPOINT,value=1e-7)
"""

    lexer = MysfireHeaderLexer()
    parser = MysfireHeaderParser()
    for line in _PARSER_TEST_DATA.splitlines():
        if not line:
            continue
        for processor in parser.parse(lexer.tokenize(line)):
            print(processor)
            print("--")
