//! Parser(Tokens -> Abstract Syntax Tree)

use std::iter::Peekable;

use inkwell::support::enable_llvm_pretty_stack_trace;

use crate::ast::{self, Operator};
use crate::token::Token;

pub trait Parse
where
    Self: Sized,
{
    fn parse(input: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Self, String>;
}

pub trait TokenStream {
    fn parse<T: Parse>(&mut self) -> Result<T, String>;
}

impl<I: Iterator<Item = Token>> TokenStream for Peekable<I> {
    fn parse<T: Parse>(&mut self) -> Result<T, String> {
        T::parse(self)
    }
}

pub fn try_parse_expr(input: &mut Peekable<impl Iterator<Item = Token>>) -> Option<ast::Node> {
    println!("===> Parsing Expression: {:?}", input.peek());

    match input.peek() {
        Some(Token::String(s)) => Some(ast::Node::String(s.clone())),
        Some(Token::Number(num)) => Some(ast::Node::Int(*num)),
        Some(Token::Boolean(boolean)) => Some(ast::Node::Boolean(*boolean)),
        _ => None,
    }
}

pub fn parse_node(input: &mut Peekable<impl Iterator<Item = Token>>) -> Result<ast::Node, String> {
    println!("==> Parsing Node");

    input.next();
    let token = input.peek().ok_or("unexpected end of file")?;

    if let Some(operator) = ast::Operator::from_token(token.clone()) {
        let child = try_parse_expr(input).ok_or("Failed to parse child of unary operator")?;

        Ok(ast::Node::UnaryExpr {
            op: operator,
            child: Box::new(child),
        })
    } else {
        let lhs = try_parse_expr(input).ok_or("failed to parse expr")?;

        let token = input.peek().ok_or("unexpected end of file")?;

        if let Some(op) = ast::Operator::from_token(token.clone()) {
            let rhs = try_parse_expr(input).ok_or("failed to parse right side of expression")?;
            input.next();

            Ok(ast::Node::BinaryExpr {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        } else {
            input.next();
            parse_node(input)
        }
    }
}
