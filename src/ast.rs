//! Abstract Syntax Tree Types

use crate::{
    lexer::TokenLexer,
    parser::{self, TokenStream},
    token::Token,
};
use std::{any::TypeId, collections::HashMap, iter::Peekable, ops::Deref};

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

#[derive(Debug, Clone)]
pub enum Node {
    Expr(Expr),
    Stmt(Statement),
    Item(Item),
}

pub type AST = (String, Vec<Node>);

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
        ty: Type,
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

 ($target:expr, $pat:path) => {{
        match $target{
            $pat => Some(()), // #1
            _ => panic!("mismatch variant when cast to {}", stringify!($pat)), // #2
    }}};
}

#[macro_export]
macro_rules! str_or_format{
    ($str:literal) => {{$str}};
    (($fmt:literal, $($arg:expr), *)) => {{format!($fmt, $($arg),*)}};
}

#[macro_export]
macro_rules! pipeline_send{
    (
        #[$ty:ident]
        $msg:tt,
        $($note:tt),*
    ) => {{
        use colored::Colorize as _;
        crate::pipeline::COMPILER_PIPELINE.read().expect("message pipeline locked")
            .process_message(crate::pipeline::Message::$ty{
            msg: str_or_format!($msg).to_string(),
            notes: vec![
                $(format!("  {} {}", "|".blue(), str_or_format!($note))),*
            ]
        })
    }};

    (
        #[$ty:ident]
        $msg:tt
    ) => {{
        use colored::Colorize as _;
        crate::pipeline::COMPILER_PIPELINE.read().expect("message pipeline locked")
            .process_message(crate::pipeline::Message::$ty{
            msg: str_or_format!($msg).to_string(),
            notes: vec![],
        })
    }};
}

macro_rules! error_and_return {
    ($($passed:tt)*) => {{
        pipeline_send!($($passed)*);
        return None;
    }};
}

#[inline(always)]
pub fn expect_token(input: &mut impl Iterator<Item = Token>, token: Token) -> Option<()> {
    let tok = input.next()?;

    None
}

macro_rules! try_cast {
    ($target:expr, $pat:path, $($value:ident),*) => {{
        match $target{
            $pat($($value),*) => Some(($($value),*)), // #1
            _ => None, // #2
    }}};


    ($target:expr, $pat:path) => {{
        match $target{
            $pat => Some(()), // #1
            _ => None, // #2
    }}};
}

pub fn parse_cast(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<Expr> {
    // Syntax: cast[<type>] <expr>
    input.next();

    match input.next()? {
        Token::OpeningBracket => {}
        t => {
            pipeline_send!(
                #[Error]
                "Unexpected token in expression.",
                ("Expected opening bracket found \"{:?}\".", t)
            );
            return None;
        }
    };

    let target_ty = type_from_str(try_cast!(input.next()?, Token::Identifier, ty)?, ctx)?;

    match input.next()? {
        Token::ClosingBracket => {}
        t => {
            pipeline_send!(
                #[Error]
                "Unexpected token in expression.",
                ("Expected closing bracket found \"{:?}\".", t)
            );
            return None;
        }
    };

    let expr = Expr::parse(input, ctx)?;

    Some(Expr::Cast {
        expr: Box::new(expr),
        into: target_ty,
    })
}

impl Expr {
    pub fn parse(
        input: &mut Peekable<impl Iterator<Item = Token>>,
        ctx: &mut Context,
    ) -> Option<Self> {
        let token = input.peek()?;

        if token == &Token::LeftParenthesis {
            input.next();
            let res = Self::parse(input, ctx)?;
            input.next();
            return Some(res);
        }

        match_exprs! {
                 Value::new(token) => Some(value) => {
                     input.next()?;
                     return Some(if let Some(op) = Operator::from_token(input.peek().map(|v|v.clone())?){
                         input.next()?;
                         Self::Binary { op, rhs: Box::new(Self::Constant(value)), lhs: Box::new(Self::parse(input, ctx)?)}
                     }else {

                         Self::Constant(value)
                     });
                 },
                 Operator::from_token(token.clone()) => Some(op) => {
                     input.next()?;
                     return Some(Self::Unary { op, child: Box::new(Self::parse(input, ctx)?) });
                 },
                 token => Token::Cast => return parse_cast(input, ctx),
                token => Token::Identifier(ident) => {
                    let ident_cloned = ident.clone();
                    let ty = ctx.typeof_variable(ident_cloned.clone())?;
                    input.next()?;
                    return Some(Self::Variable { ty, ident: ident_cloned }    );
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
        Expr::Variable { ty, .. } => ty,
        Expr::Cast { into, .. } => into,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Type {
    Boolean,
    UnsignedInt {
        bits: u8,
    },
    Int {
        bits: u8,
    },
    String,
    Float {
        bits: u8,
    },
    Array {
        len: usize,
        item_ty: Box<Type>,
    },
    Pointee {
        target_ty: Box<Type>,
    },
    Void,
    Never,
    #[default]
    Unknown,
    _Custom(Box<Type>),
    _Builtin,
}

pub fn type_from_str(s: String, ctx: &mut Context) -> Option<Type> {
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
        "__flakec_builtin" => Some(Type::_Builtin),
        _ => match ctx.try_get_type(s.clone()) {
            Some(ty) => return Some(Type::_Custom(Box::new(ty))),
            None => {
                pipeline_send!(
                    #[Error]
                    ("Type {} not fond.", s.clone()),
                    "Consider adding a type alias."
                );
                None
            }
        },
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum Item {
    TypeAlias {
        name: String,
        target: Type,
    },
    Module {
        name: String,
        is_transparent: bool,
        content: Vec<Node>,
    },
    GlobalVariable {
        name: String,
        ty: Type,
    },
}

impl Item {
    pub fn parse(
        input: &mut Peekable<impl Iterator<Item = Token>>,
        ctx: &mut Context,
    ) -> Option<Item> {
        let token = input.peek()?;

        match token {
            Token::TypeAlias => return parse_type_alias(input, ctx),
            _ => return None,
        }
    }
}

pub fn parse_raw_params(input: &mut Peekable<impl Iterator<Item = Token>>) -> Option<Vec<Token>> {
    // Raw Param Syntax:  [ <...> ]

    if try_cast!(input.peek()?, Token::OpeningBracket).is_some() {
        input.next();
    } else {
        return None;
    }

    let mut brackets = 1;
    let mut collected = Vec::<Token>::new();

    loop {
        if brackets == 0 {
            break;
        }

        let token = input.peek()?;

        match token {
            Token::OpeningBracket => {
                brackets += 1;
            }
            Token::ClosingBracket => {
                brackets -= 1;
            }
            Token::EOF => error_and_return!(
                #[Error]
                "Unexpected end of file",
                "Have you forgotten a semicolon?"
            ),
            _ => collected.push(token.clone()),
        };

        input.next();
    }

    Some(collected)
}

pub fn parse_type<'a>(input: &mut impl Iterator<Item = Token>, ctx: &mut Context) -> Option<Type> {
    let mut input = input;
    let ident = match try_cast!(input.next()?, Token::Identifier, ident) {
        Some(ident) => ident,
        None => error_and_return!(
            #[Error]
            "Expected type identfier.",
            "Types need to be valid identfiers!"
        ),
    };

    return match type_from_str(ident.clone().clone(), ctx) {
        Some(ty) => Some(ty),
        None => error_and_return!(
            #[Error]
            (
                "the type \"{}\" cannot be found in the current context",
                ident
            ),
            "Consider adding a type alias."
        ),
    };
}

pub fn parse_let(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<Statement> {
    // Syntax:
    // - let <name> = <value>; <-- Implicit Type
    // - let[<type>] <name> = <value> <-- Explicit Type

    input.next()?; // Skip [Let] token.

    let maybe_explicit_type = parse_raw_params(input);

    let ident = try_cast!(input.next()?, Token::Identifier, ident)?;

    _ = try_cast!(input.next()?, Token::Equals)?;

    let value = Expr::parse(input, ctx)?;

    _ = try_cast!(input.next()?, Token::Semicolon)?;

    let ty = match maybe_explicit_type {
        Some(toks) => parse_type(&mut toks.iter().cloned(), ctx)?,
        None => match infer_expr_type(value.clone()) {
            Type::Unknown => error_and_return!(
                #[Error]
                "the type cannot be infered, consider using casting."
            ),
            ty => ty,
        },
    };

    ctx.register_local_variable(ident.clone(), ty.clone() as VarMeta);

    Some(Statement::Let {
        ty,
        ident,
        value: value.clone(),
    })
}

pub type VarMeta = Type;

#[derive(Debug, Clone)]
pub struct Context {
    pub locals: HashMap<String, VarMeta>,
    pub can_return: bool,
    pub types: HashMap<String, Type>,
}

impl Context {
    pub fn typeof_variable(&mut self, ident: String) -> Option<Type> {
        let meta = self.locals.get(&ident)?;

        Some(meta.clone() as Type)
    }

    pub fn try_get_type(&mut self, name: String) -> Option<Type> {
        self.types.get(&name).cloned()
    }

    pub fn register_local_variable(&mut self, name: String, ty: Type) -> Option<Type> {
        self.locals.insert(name, ty)
    }

    pub fn register_type_alias(&mut self, name: String, target: Type) -> Option<Type> {
        self.types.insert(name, target)
    }
}

pub fn parse_type_alias(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<Item> {
    input.next()?; // Skip "type" token.

    let ident = try_cast!(input.next()?, Token::Identifier, ident)?;

    if try_cast!(input.next()?, Token::Equals).is_none() {
        error_and_return!(
            #[Error]
            "expcted equals sign."
        );
    };

    let target = parse_type(input, ctx)?;

    _ = try_cast!(input.next()?, Token::Semicolon)?;

    ctx.register_type_alias(ident.clone(), target.clone());

    Some(Item::TypeAlias {
        name: ident,
        target,
    })
}

impl Statement {
    pub fn parse(
        input: &mut Peekable<impl Iterator<Item = Token>>,
        ctx: &mut Context,
    ) -> Option<Self> {
        let token = input.peek()?;

        match_exprs! {
            token => Token::Let => {
               //  input.next();
                return parse_let(input, ctx);
            },
            token => Token::Return => {
                input.next()?;
                match ctx.can_return{
                    true => {
                        let res = Some(Self::Return{ value: Expr::parse(input, ctx)?});
                        _ = cast!(input.next()?, Token::Semicolon);
                        return res;
                    },
                    false => error_and_return!(
                        #[Error] // <-- Not really a syntax error!
                        "the \"return\" statement isn't allowed in this context."
                    )
                }
            }
        };

        None
    }
}

pub fn parse_node(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<Node> {
    match_exprs! {
        Item::parse(input, ctx) => Some(item) => return Some(Node::Item(item)),
        Statement::parse(input, ctx) => Some(stmt) => return  Some(Node::Stmt(stmt)),
        Expr::parse(input, ctx) => Some(expr) => return Some(Node::Expr(expr))
    }

    None
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
