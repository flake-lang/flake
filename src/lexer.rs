use std::iter::Peekable;

use crate::token::{self, Token};

#[derive(Debug)]
#[repr(u32)]
pub enum LexError {
    InvalidCharacter,
    UnexpectedEndOfFile,
    InvalidStringLiteral,
    IntegerParsingFailedi,
}

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
            _ => None,
        }
    }
}

pub fn lex_token(token: Token, input: &mut Peekable<impl Iterator<Item = char>>) -> Token {
    let mut current_token = token;
    input.next();

    loop {
        let chr = peek_or_break!(input);

        if let Some(combined_token) = current_token.try_combine_with(*chr) {
            input.next();
            current_token = combined_token;
        } else {
            break;
        }
    }

    current_token
}

pub fn try_lex_keyword(s: String) -> Option<Token> {
    match s.as_str() {
        "let" => Some(Token::Let),
        "return" => Some(Token::Return),
        "cast" => Some(Token::Cast),
        "true" => Some(Token::Boolean(true)),
        "false" => Some(Token::Boolean(false)),
        "type" => Some(Token::TypeAlias),
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
) -> Result<i64, LexError> {
    let mut buffer = String::new();
    let mut maybe_sign_possible = true;

    loop {
        let chr = peek_or_break!(input);
        match chr {
            '0'..'9' => {
                let c = chr.clone();
                input.next();
                buffer.push(c);
                maybe_sign_possible = false;
            }
            '-' => {
                if maybe_sign_possible {
                    input.next();
                    buffer.push('-');
                    maybe_sign_possible = false;
                }
            }
            _ => break,
        };
    }

    Ok(buffer
        .parse()
        .map_err(|_| LexError::IntegerParsingFailedi)?)
}

pub fn create_lexer<'a>(code: &'a str) -> impl Iterator<Item = token::Token> + 'a {
    std::iter::from_coroutine(move || {
        let mut input = code.chars().peekable();

        loop {
            let chr = peek_or_break!(input);

            match chr {
                ' ' | '\n' => {
                    input.next();
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
                '[' => yield lex_token(Token::OpeningBracket, &mut input),
                ']' => yield lex_token(Token::ClosingBracket, &mut input),
                '{' => yield lex_token(Token::LeftBrace, &mut input),
                '}' => yield lex_token(Token::RightBrace, &mut input),
                '"' => {
                    yield Token::String(
                        lex_string_literal(&mut input).expect("Failed to lex string literal"),
                    )
                }
                '0'..'9' => {
                    yield Token::Number(
                        lex_number_literal(&mut input).expect("Failed to lex number literal"),
                    )
                }
                _ => {
                    yield match lex_unhandled(&mut input) {
                        Token::Identifier(ident) => {
                            if ident.is_empty() {
                                break;
                            } else {
                                Token::Identifier(ident)
                            }
                        }
                        t => t,
                    }
                }
            }
        }

        yield Token::EOF;
    })
}
