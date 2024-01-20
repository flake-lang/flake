//! Abstract Syntax Tree Types

use crate::{parser, token::Token};
use std::iter::Peekable;

#[derive(Debug, Clone, Copy)]
pub enum Operator {
    Plus,
    Minus,
    Divide,
    Multiply,
    Modulo,
}

impl Operator {
    pub fn from_token(token: Token) -> Option<Self> {
        match token {
            Token::Plus => Some(Self::Plus),
            Token::Minus => Some(Self::Minus),
            Token::Slash => Some(Self::Divide),
            Token::Star => Some(Self::Multiply),
            Token::Percent => Some(Self::Modulo),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum Node {
    Int(i64),
    Boolean(bool),
    String(String),
    UnaryExpr {
        op: Operator,
        child: Box<Node>,
    },
    BinaryExpr {
        op: Operator,
        lhs: Box<Node>,
        rhs: Box<Node>,
    },
}
