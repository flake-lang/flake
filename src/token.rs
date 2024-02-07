#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
#[non_exhaustive]
pub enum TokenKind {
    Plus,
    Minus,
    Star,
    Percent,
    Slash,
    Equals,
    DoubleEquals,
    NotEquals,
    ExclamationMark,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Assignment,
    String(String),
    Number(i64),
    Identifier(String),
    Boolean(bool),
    And,
    Or,
    Not,
    LeftParenthesis,
    RightParenthesis,
    LeftBrace,
    OpeningBracket,
    ClosingBracket,
    RightBrace,
    Comma,
    Semicolon,
    Let,
    At,
    DoubleAt,
    Return,
    Cast,
    TypeAlias,
    Import,
    Mod,
    EOF,
    /// Example: generic.ext-token.sized
    _ViaIdent(String),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokeniNew {
    kind: TokenKind,
    span: Span,
}

pub type Token = TokenKind;

#[derive(Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl core::fmt::Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("#({}..{})", self.start, self.end))
    }
}

pub struct TokenStream<'a, I: Iterator<Item = Token>>(&'a mut std::iter::Peekable<I>);

impl<'a, I: Iterator<Item = Token>> Iterator for TokenStream<'a, I> {
    type Item = TokenKind;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
