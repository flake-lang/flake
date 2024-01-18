//! Abstract Syntax Tree Types

use crate::token::Token;
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

pub fn parse_node(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Node, String> {
    let token = tokens.peek().ok_or("end of file")?.clone();

    match token {
        Token::LeftParenthesis => {
            tokens.next();
            let tmp = parse_node(tokens)?;
            tokens.next();
            return Ok(tmp);
        }
        Token::Number(left) => {
            tokens.next();
            let operator = {
                let next_token = tokens.peek().ok_or("end of file".to_owned())?;
                if let Some(operator) = Operator::from_token(next_token.clone()) {
                    tokens.next();
                    operator
                } else {
                    tokens.next();
                    return Ok(Node::Int(left));
                }
            };

            let rhs = parse_node(tokens)?;

            return Ok(Node::BinaryExpr {
                op: operator,
                lhs: Box::new(Node::Int(left)),
                rhs: Box::new(rhs),
            });
        }
        _ => Err(format!("Failed to parse node! Invalid Token: {:#?}", token)),
    }
}
