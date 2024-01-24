//! Unit Tests

/// Tests related to Flake's Lexer.
pub mod lexer {
    use crate::lexer::*;
    use crate::token::Token::{self, *};

    #[test]
    pub fn string_literalsl() {
        let tokens = create_lexer("\"Hello World!\"").collect::<Vec<Token>>();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], Token::String("Hello World!".to_owned()));
        assert_eq!(tokens[1], Token::EOF);
    }

    #[test]
    pub fn numeric_literals() {
        let tokens = create_lexer("123").collect::<Vec<Token>>();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], Token::Number(123));
        assert_eq!(tokens[1], Token::EOF);
    }

    #[test]
    pub fn binary_operations() {
        let tokens = create_lexer("10 - 2").collect::<Vec<Token>>();

        assert_eq!(tokens.as_slice(), [Number(10), Minus, Number(2), EOF]);
    }

    #[test]
    pub fn keywords() {
        let tokens = create_lexer("return;").collect::<Vec<Token>>();

        assert_eq!(tokens.as_slice(), [Return, Semicolon, EOF]);
    }

    #[test]
    pub fn identifiers() {
        let tokens = create_lexer("example_identifier_123").collect::<Vec<Token>>();

        assert_eq!(
            tokens.as_slice(),
            [Identifier("example_identifier_123".to_owned()), EOF]
        )
    }
}

/*
mod parser {
    use crate::parser::*;

    #[test]
    pub fn unary_expressions() {
        todo!()
    }

    #[test]
    pub fn binary_expressions() {
        todo!()
    }

    #[test]
    pub fn operators() {
        todo!()
    }

    #[test]
    pub fn constant_values() {
        todo!()
    }
}
*/
