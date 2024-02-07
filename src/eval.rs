use std::{
    collections::HashMap,
    ops::{Deref, DerefMut, Neg},
};

use crate::{
    ast::{Expr, Operator, Statement, Type, Value},
    token::{Token, TokenKind},
};

#[derive(Clone, Debug)]
pub struct Context {
    pub variables: HashMap<String, Value>,
}

pub fn eval_expr(expr: Expr, ctx: &mut Context) -> Option<Value> {
    println!("Evaluating...");
    match expr {
        Expr::Constant(v) => return Some(v),
        Expr::Variable { ty, ident } => return ctx.variables.get(&ident).cloned(),
        Expr::Cast { into, expr } => eval_expr(*expr, ctx),
        Expr::Unary { op, child } => eval_unary(op, eval_expr(*child, ctx)?),
        Expr::Binary { op, rhs, lhs } => {
            exec_operation(op, eval_expr(*rhs, ctx)?, eval_expr(*lhs, ctx)?)
        }
        _ => unimplemented!(),
    }
}

pub fn eval_unary(op: Operator, value: Value) -> Option<Value> {
    match (op, value.get_type()) {
        (Operator::Not, Type::Boolean) => match *value {
            TokenKind::Boolean(val) => Value::new(&TokenKind::Boolean(!val)),
            _ => unimplemented!(),
        },
        (Operator::Minus, Type::UnsignedInt { .. }) => match *value {
            TokenKind::Number(n) => Value::new(&TokenKind::Number(-n)),
            _ => unimplemented!(),
        },
        _ => todo!(),
    }
}

pub fn eval_statement(stmt: Statement, ctx: &mut Context) -> Option<()> {
    match stmt {
        Statement::Let { ty, ident, value } => {
            let val = eval_expr(value, ctx)?;
            ctx.variables.insert(ident, val);
        }
        _ => todo!(),
    };

    Some(())
}

pub fn exec_operation(op: Operator, rhs: Value, lhs: Value) -> Option<Value> {
    Value::new(&{
        match (op, rhs.deref(), lhs.deref()) {
            (Operator::Plus, TokenKind::Number(n1), TokenKind::Number(n2)) => {
                TokenKind::Number(n1 + n2)
            }
            (Operator::Minus, TokenKind::Number(n1), TokenKind::Number(n2)) => {
                Token::Number(n1 - n2)
            }
            (Operator::Eq, ..) => TokenKind::Boolean(cmp_values(rhs, lhs)),
            _ => todo!(),
        }
    })
}

fn cmp_values(v1: Value, v2: Value) -> bool {
    use TokenKind::*;
    match (v1.deref(), v2.deref()) {
        (Number(n), Number(n2)) => n == n2,
        (Boolean(b), Boolean(b2)) => b == b2,
        (String(s), String(s2)) => s == s2,
        _ => todo!(),
    }
}
