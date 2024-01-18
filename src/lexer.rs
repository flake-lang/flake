
use std::iter::Peekable;

use crate::token::{self, Token};

#[derive(Debug)]
#[repr(u32)]
enum LexError{
    InvalidCharacter,
    UnexpectedEndOfFile,
    InvalidStringLiteral,
    IntegerParsingFailedi,
}

pub trait TokenLexer{
//  fn process(input: &mut Peekable<impl Iterator<Item = char>>) -> Self;
    fn try_combine_with(&self, chr: char) -> Option<Token>;
}

impl TokenLexer for Token{
    fn try_combine_with(&self, chr: char) -> Option<Token>{
        match (self, chr){
            (Token::String(_), _) |
            (Token::Number(_), _) => unimplemented!(),
            (Token::Equals, '=') => Some(Token::DoubleEquals),
            _ => None,
        }
    }
}

pub fn lex_token(token: Token, input: &mut Peekable<impl Iterator<Item = char>>) -> Token{
    let mut current_token = token;
    input.next();
    
    loop{
        let chr = if let Some(chr) = input.peek(){
            chr
        }else {
            break;
        };

        if let Some(combined_token) = current_token.try_combine_with(*chr){
            input.next();
            current_token = combined_token;
        }else{
            break;
        }
    };

    current_token
} 

pub fn lex_string_literal(input: &mut Peekable<impl Iterator<Item = char>>) -> Result<String, LexError> {
    input.next();

    let mut buffer = String::new();

    loop {
        let chr = input.next().unwrap();
        match chr{
            '"' => break,
            _  => buffer.push(chr)
        }
    } 

    Ok(buffer)
}

pub fn lex_number_literal(input: &mut Peekable<impl Iterator<Item = char>>) -> Result<i64, LexError>{
    let mut buffer = String::new();
    let mut maybe_sign_possible = true;
        
    loop{
        let chr = if let Some(chr) = input.peek(){
            chr
        }else{
            break;
        };

        match chr{
            '0'..'9' => {
                let c = chr.clone();
                input.next();
                buffer.push(c);
                maybe_sign_possible = false;
            },
            '-' => if maybe_sign_possible{
                input.next();
                buffer.push('-');
                maybe_sign_possible = false;      
            },
            _ => break
        };
    }

    Ok(buffer.parse().map_err(|_|LexError::IntegerParsingFailedi)?)
}

pub fn create_lexer<'a>(code: &'a str) -> impl Iterator<Item = token::Token> + 'a{
    std::iter::from_coroutine(move ||{
        let mut input = code.chars().peekable();

        loop{
            let chr = if let Some(chr) = input.peek() {
                chr
            }else{
                break;
            };

            match chr{
                ' ' => {
                    input.next();
                    continue;
                },
                '+' => yield lex_token(Token::Plus, &mut input),
                '=' => yield lex_token(Token::Equals, &mut input),
                '-' => yield lex_token(Token::Minus, &mut input),
                '*' => yield lex_token(Token::Star, &mut input),
                '%' => yield lex_token(Token::Percent, &mut input),
                '/' => yield lex_token(Token::Slash,  &mut input),
                ',' => yield lex_token(Token::Comma, &mut input),
                ';' => yield lex_token(Token::Semicolon, &mut input),
                '(' => yield lex_token(Token::LeftParenthesis, &mut input),
                ')' => yield lex_token(Token::RightParenthesis, &mut input),
                '>' => yield lex_token(Token::GreaterThan, &mut input),
                '<' => yield lex_token(Token::LessThan, &mut input),
                '{' => yield lex_token(Token::LeftBrace, &mut input),
                '}' => yield lex_token(Token::RightBrace, &mut input),
                '"' => yield Token::String(lex_string_literal(&mut input).expect("Failed to lex string literal")),
                '0'..'9' => yield Token::Number(lex_number_literal(&mut input).expect("Failed to lex number literal")),
                _ => panic!("LEX001: invalid character in source. {}", chr) // TODO: Identifiers
            }
        }

        yield Token::EOF;        
        
    })
}
