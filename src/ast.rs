//! Abstract Syntax Tree Types

use crate::{lexer::TokenLexer, parser, token::Token};
use std::{any::TypeId, iter::Peekable, ops::Deref};

#[derive(Debug, Clone, Copy)]
pub enum Operator {
    Plus,
    Minus,
    Divide,
    Multiply,
    Modulo,
    Not,
    Eq,
}

impl Operator {
    pub fn from_token(token: Token) -> Option<Self> {
        match token {
            Token::Plus => Some(Self::Plus),
            Token::Minus => Some(Self::Minus),
            Token::Slash => Some(Self::Divide),
            Token::Star => Some(Self::Multiply),
            Token::Percent => Some(Self::Modulo),
            Token::ExclamationMark => Some(Self::Not),
            Token::DoubleEquals => Some(Self::Eq),
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

#[derive(Debug, Clone)]
pub enum Expr {
    Constant(Value),
    Binary {
        op: Operator,
        rhs: Box<Expr>,
        lhs: Box<Expr>,
    },
    Unary {
        op: Operator,
        child: Box<Expr>,
    },
    Variable {
        ident: String,
    },
    Cast {
        into: Type,
        expr: Box<Expr>,
    },
}

macro_rules! match_exprs {
    {$($exp:expr => $n:pat => $on:expr),*} => {
        $(
        match $exp{
            $n => $on,
            _ => {}
            }
    );*
    };
}

macro_rules! cast {
    ($target:expr, $pat:path, $($value:ident),*) => {{
        match $target{
            $pat($($value),*) => ($($value),*), // #1
            _ => panic!("mismatch variant when cast to {}", stringify!($pat)), // #2
    }}};
}

macro_rules! try_cast {
    ($target:expr, $pat:path, $($value:ident),*) => {{
        match $target{
            $pat($($value),*) => Some(($($value),*)), // #1
            _ => None, // #2
    }}};
}

pub fn parse_cast(input: &mut Peekable<impl Iterator<Item = Token>>) -> Option<Expr> {
    // Syntax: cast[<type>] <expr>
    input.next();

    match input.next()? {
        Token::OpeningBracket => {}
        t => {
            eprintln!("error: expected opening bracke, found {:?}", t);
            return None;
        }
    };

    let target_ty = type_from_str(try_cast!(input.next()?, Token::Identifier, ty)?)?;

    match input.next()? {
        Token::ClosingBracket => {}
        _ => {
            eprintln!("error: expected closing bracket.");
            return None;
        }
    };

    let expr = Expr::parse(input)?;

    Some(Expr::Cast {
        expr: Box::new(expr),
        into: target_ty,
    })
}

impl Expr {
    pub fn parse(input: &mut Peekable<impl Iterator<Item = Token>>) -> Option<Self> {
        let token = input.peek()?;

        dbg!(&token);

        if token == &Token::LeftParenthesis {
            input.next();
            let res = Self::parse(input)?;
            input.next();
            return Some(res);
        }

        match_exprs! {
                 Value::new(token) => Some(value) => {
                     input.next()?;
                     return Some(if let Some(op) = Operator::from_token(input.peek().map(|v|v.clone())?){
                         input.next()?;
                         Self::Binary { op, rhs: Box::new(Self::Constant(value)), lhs: Box::new(Self::parse(input)?)}
                     }else {
                         Self::Constant(value)
                     });
                 },
                 Operator::from_token(token.clone()) => Some(op) => {
                     input.next()?;
                     return Some(Self::Unary { op, child: Box::new(Self::parse(input)?) });
                 },
                 token => Token::Cast => return parse_cast(input),
                 token => Token::Identifier(ident) => {
                    let ident_cloned = ident.clone();
                    input.next()?;
                    return Some(Self::Variable { ident: ident_cloned }    );
            }
        };

        //  {input.next(); Self::parse(input)} => Some(expr) => return Some(expr)

        None
    }
}

pub fn infer_expr_type(expr: Expr) -> Type {
    match expr {
        Expr::Constant(value) => value.get_type(),
        Expr::Binary { rhs, lhs, .. } => {
            let right_ty = infer_expr_type(*rhs.clone());
            let left_ty = infer_expr_type(*lhs.clone());

            if right_ty != left_ty {
                Type::Unknown
            } else {
                left_ty
            }
        }
        Expr::Unary { child, .. } => infer_expr_type(*child.clone()),
        Expr::Variable { .. } => Type::Unknown,
        Expr::Cast { into, .. } => into,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Boolean,
    UnsignedInt { bits: u8 },
    Int { bits: u8 },
    String,
    Float { bits: u8 },
    Array { len: usize, item_ty: Box<Type> },
    Pointee { target_ty: Box<Type> },
    Void,
    Never,
    Unknown,
}

pub fn type_from_str(s: String) -> Option<Type> {
    match s.as_str() {
        "u32" => Some(Type::UnsignedInt { bits: 32 }),
        "u16" => Some(Type::UnsignedInt { bits: 16 }),
        "u8" => Some(Type::UnsignedInt { bits: 8 }),
        "bool" => Some(Type::Boolean),
        "str" => Some(Type::String),
        "void" => Some(Type::Void),
        "f32" => Some(Type::Float { bits: 32 }),
        "never" => Some(Type::Never),
        "__flakec_unknown" => Some(Type::Unknown),
        "str" => Some(Type::String),
        "void" => Some(Type::Void),
        "f32" => Some(Type::Float { bits: 32 }),
        "never" => Some(Type::Never),
        "__flakec_unknown" => Some(Type::Unknown),
        _ => None,
    }
}

pub enum Statement {
    Return {
        value: Expr,
    },
    Let {
        ty: Type,
        ident: String,
        value: Expr,
    },
}

#[derive(Clone, Debug)]
pub struct Value(Token);

impl Value {
    /// Crates a new value from a token.
    pub fn new(token: &Token) -> Option<Self> {
        if Self::valid(token) {
            Some(Self(token.clone()))
        } else {
            None
        }
    }

    /// Checks if a token is a valid value.
    fn valid(token: &Token) -> bool {
        match token {
            Token::Number(_) | Token::String(_) | Token::Boolean(_) => true,
            _ => false,
        }
    }

    pub fn get_type(&self) -> Type {
        match **self {
            Token::Number(_) => Type::UnsignedInt { bits: 32 },
            Token::String(_) => Type::String,
            Token::Boolean(_) => Type::Boolean,
            _ => Type::Unknown,
        }
    }
}

impl Deref for Value {
    type Target = Token;

    fn deref(&self) -> &Token {
        if !Self::valid(&self.0) {
            unreachable!("The inner Token of a value must always be valid");
        }

        &self.0
    }
}
