use std::iter::Peekable;

use itertools::{cons_tuples, Itertools};

use crate::token::{self, Token as FullToken, TokenKind as Token};

#[derive(Debug)]
#[repr(u32)]
pub enum LexError {
    InvalidCharacter,
    UnexpectedEndOfFile,
    InvalidStringLiteral,
    IntegerParsingFailedi,
}

#[macro_export]
macro_rules! peek_or_break {
    ($peekable:ident) => {
        if let Some(value) = $peekable.peek() {
            value
        } else {
            break;
        }
    };
}

pub trait TokenLexer {
    //  fn process(input: &mut Peekable<impl Iterator<Item = char>>) -> Self;
    fn try_combine_with(&self, chr: char) -> Option<Token>;
}

impl TokenLexer for Token {
    fn try_combine_with(&self, chr: char) -> Option<Token> {
        match (self, chr) {
            (Token::String(_), _) | (Token::Number(_), _) => unimplemented!(),
            (Token::Equals, '=') => Some(Token::DoubleEquals),
            (Token::At, '@') => Some(Token::DoubleAt),
            (Token::Identifier(ident), '!') => Some(Token::Call(ident.clone())),
            _ => None,
        }
    }
}

pub fn lex_token(token: Token, input: &mut Peekable<impl Iterator<Item = char>>) -> Token {
    let mut current_token = token.clone();

    let mut is_start = true;

    match token {
        Token::Identifier(_) => {}
        _ => {
            input.next();
        }
    };

    loop {
        let chr = peek_or_break!(input);

        if let Some(combined_token) = current_token.try_combine_with(*chr) {
            input.next();
            current_token = combined_token;
            is_start = false;
        } else {
            break;
        }
    }

    current_token
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImportPath {
    file: String,
    sub_modules: Vec<String>,
    is_wildcard: bool,
}

impl ImportPath {
    pub fn parse(s: String) -> Result<Self, String> {
        if s.is_empty() {
            return Err("empty paths in imports aren't allowed.".to_owned());
        };

        if !s.contains('.') {
            return Ok(Self::with_file_unchecked(s));
        };

        let modules = s.split('.').collect::<Vec<_>>();

        if !(modules.len() >= 2) {
            return Err("path seperator isn't followed by segment.".to_owned());
        };

        let file = modules[0].to_owned();
        let mut sub_modules = modules
            .iter()
            .skip(1)
            .map(|s| (*s).to_owned())
            .collect::<Vec<_>>();

        if sub_modules.len() < 1 {
            return Err("expected sub module or wildcard.".to_owned());
        };

        let mut is_wildcard = false;
        if *sub_modules.last().unwrap() == "*".to_owned() {
            sub_modules.pop();
            is_wildcard = true;
        };

        Ok(Self {
            file,
            sub_modules,
            is_wildcard,
        })
    }

    pub fn with_file_unchecked(file: String) -> Self {
        Self {
            file,
            sub_modules: vec![],
            is_wildcard: false,
        }
    }
}

pub fn try_lex_keyword(s: String) -> Option<Token> {
    match s.as_str() {
        "let" => Some(Token::Let),
        "return" => Some(Token::Return),
        "cast" => Some(Token::Cast),
        "__flakec_eval" => Some(Token::Comptime),
        "mod" => Some(Token::Mod),
        "fn" => Some(Token::Function),
        "true" => Some(Token::Boolean(true)),
        "false" => Some(Token::Boolean(false)),
        "import" => Some(Token::Import),
        "type" => Some(Token::TypeAlias),
        "__flakec_inlined_block" => Some(dbg!(Token::_ViaIdent(
            "internal.keyword.inlined-block".to_owned()
        ))),
        _ => None,
    }
}

pub fn lex_unhandled(input: &mut Peekable<impl Iterator<Item = char>>) -> Token {
    let raw = {
        let mut buffer = String::new();

        loop {
            let chr = peek_or_break!(input);

            match chr {
                'A'..'Z' | 'a'..'z' | '_' | '0'..'9' => buffer.push(*chr),
                _ => break,
            }

            input.next();
        }

        buffer
    };

    match try_lex_keyword(raw.clone()) {
        Some(keyword) => keyword,
        None => Token::Identifier(raw),
    }
}

pub fn lex_string_literal(
    input: &mut Peekable<impl Iterator<Item = char>>,
) -> Result<String, LexError> {
    input.next();

    let mut buffer = String::new();

    loop {
        let chr = input.next().unwrap();
        match chr {
            '"' => break,
            _ => buffer.push(chr),
        }
    }

    Ok(buffer)
}

pub fn lex_number_literal(
    input: &mut Peekable<impl Iterator<Item = char>>,
) -> Result<u64, LexError> {
    let mut buffer = String::new();
    let mut maybe_sign_possible = true;

    loop {
        let chr = peek_or_break!(input).clone();
        if chr.is_ascii_digit() {
            input.next();
            buffer.push(chr);
        } else {
            break;
        }
    }

    Ok(buffer
        .parse()
        .map_err(|_| LexError::IntegerParsingFailedi)?)
}

struct LexerInput<I: Iterator<Item = char>> {
    pub inner: I,
    pub position: usize,
    pub segment_start: usize,
}

impl<'a, I: Iterator<Item = char>> Iterator for LexerInput<I> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.position += 1;
        self.inner.next()
    }
}

pub fn lex_char_literal(
    input: &mut Peekable<impl Iterator<Item = char>>,
) -> Result<char, LexError> {
    let lit = input.take(3);

    match lit.collect::<Vec<char>>().as_slice() {
        ['\'', c, '\''] => Ok(*c),
        _ => Err(LexError::InvalidCharacter),
    }
}

pub fn create_lexer<'a>(code: &'a str) -> impl Iterator<Item = token::TokenKind> + 'a {
    std::iter::from_coroutine(move || {
        let mut stream = code.chars().into_iter();
        let mut input = stream.peekable();

        let mut is_comment = false;

        loop {
            let chr = peek_or_break!(input).clone();

            match (is_comment, chr.clone()) {
                (true, '\n') => {
                    is_comment = false;
                    input.next();
                    continue;
                }
                (true, _) => {
                    input.next();
                    continue;
                }
                _ => {}
            }

            if chr == '/' {
                if input.peek() == Some(&'/') {
                    is_comment = true;
                    continue;
                }
            }

            match chr {
                ' ' | '\t' => {
                    input.next();
                    continue;
                }
                '\n' => {
                    input.next();
                    crate::pipeline::COMPILER_PIPELINE
                        .read()
                        .expect("message pipeline locked")
                        .current_line
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
                    continue;
                }
                '+' => yield lex_token(Token::Plus, &mut input),
                '=' => yield lex_token(Token::Equals, &mut input),
                '-' => yield lex_token(Token::Minus, &mut input),
                '*' => yield lex_token(Token::Star, &mut input),
                '%' => yield lex_token(Token::Percent, &mut input),
                '!' => yield lex_token(Token::ExclamationMark, &mut input),
                '/' => yield lex_token(Token::Slash, &mut input),
                ',' => yield lex_token(Token::Comma, &mut input),
                ';' => yield lex_token(Token::Semicolon, &mut input),
                '(' => yield lex_token(Token::LeftParenthesis, &mut input),
                ')' => yield lex_token(Token::RightParenthesis, &mut input),
                '>' => yield lex_token(Token::GreaterThan, &mut input),
                '<' => yield lex_token(Token::LessThan, &mut input),
                '$' => yield lex_token(Token::Dollar, &mut input),
                '@' => yield lex_token(Token::At, &mut input),
                '&' => yield lex_token(Token::And, &mut input),
                '|' => yield lex_token(Token::Or, &mut input),
                '[' => yield lex_token(Token::OpeningBracket, &mut input),
                ':' => yield lex_token(Token::Colon, &mut input),
                ']' => yield lex_token(Token::ClosingBracket, &mut input),
                '{' => yield lex_token(Token::LeftBrace, &mut input),
                '}' => yield lex_token(Token::RightBrace, &mut input),
                '"' => {
                    yield Token::String(
                        lex_string_literal(&mut input).expect("Failed to lex string literal"),
                    )
                },
                '\'' => {
                    yield Token::Char(
                        lex_char_literal(&mut input).expect("Failed to lex char literal"),
                    )
                },
                '0'..'9' => {
                    yield Token::Number(
                        lex_number_literal(&mut input).expect("Failed to lex number literal"),
                    )
                },
                _ => {
                    yield lex_token(
                        match lex_unhandled(&mut input) {
                            Token::Identifier(ident) => {
                                if ident.is_empty() {
                                    break;
                                } else {
                                    Token::Identifier(ident)
                                }
                            }
                            t => t,
                        },
                        &mut input,
                    )
                }
            }
        }

        yield Token::EOF;
    })
}
