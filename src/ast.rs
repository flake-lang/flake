//! Abstract Syntax Tree Types

use crate::{
    feature::{FeatureGate, FeatureKind},
    lexer::{ImportPath, TokenLexer},
    parser::{self, TokenStream},
    token::Token,
};
use std::{
    any::TypeId, collections::HashMap, iter::Peekable, ops::Deref, str::Matches,
    sync::mpsc::RecvTimeoutError,
};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Node {
    Expr(Expr),
    Stmt(Statement),
    Item(Item),
    GlobalMarker { name: String, args: Vec<Value> },
}

pub type AST = (String, Vec<Node>);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        #[allow(unused_imports)]
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

#[macro_export]
macro_rules! error_and_return {
    ($($passed:tt)*) => {{
        pipeline_send!($($passed)*);
        return Default::default();
    }};
}

#[inline(always)]
pub fn expect_token(input: &mut impl Iterator<Item = Token>, token: Token) -> Option<Token> {
    let tok = input.next()?;

    match tok.clone() {
        token => Some(tok),
        _ => error_and_return!(
            #[Error]
            "Unexpected token in input.",
            ("Expected token {:?} found {:?}!", token, tok)
        ),
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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

pub macro with_feature($ctx:expr, $name:literal, {$($code:tt)*} else {$($else_code:tt)*}) {
    if $ctx.check_feature_gate($name) {
        $($code)*
    }else{
        $($else_code)*
    }
}

macro feature_err($name:literal, $($line:literal),*) {{
    use $crate::str_or_format;
    $crate::pipeline_send!(
        #[Error]
        ("{:?} is an unstable/internal feature.", $name),
        (
            "note: add \"@@feature[{:?}]\" to a module to enable it.",
            $name
        ),
        $($line),*
    );
    Default::default()
}}

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
        "__flakec_unknown" => {
            with_feature!(ctx, "type-system-internals", {Some(Type::Unknown)} else {feature_err!("type-system-internals", "required to use this item.")})
        }
        "__flakec_builtin" => with_feature!(
                ctx,
                "type-system-internals",
                {Some(Type::_Builtin)} else {feature_err!("type-system-internals", "required to use this item.")}
        ),
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum ItemMeta {
    Marker(String, Vec<Value>),
    Documentation(String),
}

pub fn split_and_parse_args_1(raw: Vec<Token>) -> Option<Vec<Value>> {
    // V1 Syntax: [<value>, ...]

    let mut values: Vec<Value> = vec![];

    let mut input = raw.iter().peekable();
    for token in input {
        if token == &Token::Comma {
            continue;
        }

        match Value::new(token) {
            Some(val) => values.push(val),
            None => error_and_return!(
                #[Error]
                ("expected constant value, found token {:?}.", token),
                "note: In V1 Argument Lists only constant are allowed, no expressions!"
            ),
        }
    }

    Some(values)
}

pub fn parse_marker(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
    global: bool,
) -> Option<(String, Vec<Value>)> {
    _ = if !global {
        try_cast!(input.peek()?, Token::At)?
    } else {
        try_cast!(input.peek()?, Token::DoubleAt)?
    };
    input.next();

    let name = try_cast!(input.next()?, Token::Identifier, ident)?;

    let args = match parse_raw_params(input) {
        Some(params) => split_and_parse_args_1(params)?,
        None => vec![],
    };

    Some((name, args))
}

pub fn parse_item_meta_continous(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<(Item, Vec<ItemMeta>)> {
    // let token = input.peek()?;

    let mut metas = Vec::<ItemMeta>::new();

    loop {
        match_exprs! {
            parse_marker(input, ctx, false) => Some(marker) => metas.push(ItemMeta::Marker(marker.0, marker.1)),
            Item::parse(input, ctx) => Some(item) => {
                return Some((item, metas));
            }
        // error_and_return!(#[Error] "Invalid meta for item.")
        }
    }

    None
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Item {
    TypeAlias {
        name: String,
        target: Type,
    },
    Module {
        name: String,
        is_transparent: bool,
        tree: Vec<Node>,
        context: Context,
    },
    Import(ImportPath),
    GlobalVariable {
        name: String,
        ty: Type,
    },
    InlinedBlock(Block),
    #[serde(untagged)]
    _Disabled,
    #[serde(skip, untagged)]
    _Global,
}

impl Item {
    pub fn parse(
        input: &mut Peekable<impl Iterator<Item = Token>>,
        ctx: &mut Context,
    ) -> Option<Item> {
        let token = input.peek()?;

        match token {
            Token::Import => return parse_import(input, ctx),
            Token::TypeAlias => return parse_type_alias(input, ctx),
            Token::Mod => return parse_mod(input, ctx),
            /* Token::_ViaIdent("internal.keyword.inlined-block") => {
                return Some(Item::InlinedBlock(parse_block(input, ctx, true)?))
            }*/
            _ => {
                pipeline_send!(
                    #[Warning]
                    "invalid item, trying to skip token."
                );
                input.next();
                return None;
            }
        }
    }
}

pub fn parse_import(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<Item> {
    // Syntax: import "compiler-builtins.primitive.*";

    _ = expect_token(input, Token::Import)?;

    let path_str = try_cast!(input.next()?, Token::String, s)?;

    let path = match ImportPath::parse(path_str) {
        Err(err) => error_and_return!(
            #[Error]
            ("{}", err)
        ),
        Ok(path) => path,
    };

    Some(Item::Import(path))
}

pub fn parse_mod(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
) -> Option<Item> {
    // Syntax: mod <name> { ... };

    _ = expect_token(input, Token::Mod)?; // ...double check... ;)

    let ident = try_cast!(input.next()?, Token::Identifier, ident)?;

    let block = parse_block(input, ctx, false)?;

    _ = expect_token(input, Token::Semicolon)?;

    Some(Item::Module {
        name: ident,
        is_transparent: false,
        tree: block.0,
        context: block.1,
    })
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
pub enum MarkerImpl {
    BuiltIn(BuiltinMarkerFunc),
}

use crate::builtins;
pub(crate) type BuiltinMarkerFunc =
    for<'item, 'ctx> fn(&'item mut Item, &'ctx mut Context, Vec<Value>);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Context {
    pub locals: HashMap<String, VarMeta>,
    pub can_return: bool,
    pub types: HashMap<String, Type>,
    pub feature_gates: HashMap<String, FeatureGate>,
    #[serde(skip)]
    pub markers: HashMap<String, MarkerImpl>,
}

impl Context {
    pub fn typeof_variable(&mut self, ident: String) -> Option<Type> {
        let meta = self.locals.get(&ident)?;

        Some(meta.clone() as Type)
    }

    pub fn apply_item_meta(&mut self, item: &mut Item, meta: ItemMeta) {
        match meta {
            ItemMeta::Marker(name, args) => {
                let marker_impl = match self.markers.get(&name) {
                    Some(_impl) => _impl,
                    None => error_and_return!(
                        #[Error]
                        (
                            "The marker {:?} doesn't exist in this context.",
                            name.clone().trim_start_matches("__global_")
                        )
                    ),
                };

                match marker_impl {
                    &MarkerImpl::BuiltIn(func) => func(item, self, args),
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }
    }

    pub fn toggle_feature_gate(&mut self, name: &'_ str) {
        let state = self.feature_gates.get(&name.to_string()).cloned();
        match state {
            Some((false, kind)) => {
                self.feature_gates
                    .insert(name.to_owned(), (true, kind.clone()));

                if kind == FeatureKind::Internal {
                    pipeline_send!(
                        #[Warning]
                        ("Use of internal feature {:?}.", name),
                        "Using Internal feature is not recommended!"
                    );
                }
            }
            Some((true, _)) => pipeline_send!(
                #[Warning]
                ("tried to enable already enabled feature {:?}.", name),
                "action: ignored"
            ),
            None => {
                pipeline_send!(
                    #[Error]
                    ("The feature {} doesn't exist.", name)
                );
                return;
            }
        };
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

    pub fn check_feature_gate(&mut self, feature: &'_ str) -> bool {
        let feature_gate = match self.feature_gates.get(&feature.to_string()) {
            Some(f) => f,
            None => {
                pipeline_send!(
                    #[Error]
                    ("The feature {:?} doesn't exist.", feature)
                );
                return false;
            }
        };

        if feature_gate.1 == FeatureKind::Internal {
            pipeline_send!(
                #[Warning]
                ("Use of internal feature {:?}.", feature),
                "Using Internal feature is not recommended!"
            );
        }

        feature_gate.0
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
        input.peek() => Some(Token::DoubleAt) => {
            let marker = parse_marker(input, ctx, true)?;

            let  marker_cloned = marker.clone();

            let mut _tmp = Item::_Global;

            Context::apply_item_meta(ctx, &mut _tmp, ItemMeta::Marker(["__global_", marker.0.as_str()].concat(), marker.1));

            return Some(Node::GlobalMarker { name: marker_cloned.0, args:  marker_cloned.1 });
        },
        parse_item_meta_continous(input, ctx) => Some((item, metas)) => {
            let mut output = item;

            for meta in metas{
                Context::apply_item_meta(ctx, &mut output, meta);
            }

            return Some(Node::Item(output))
        },
        Statement::parse(input, ctx) => Some(stmt) => return  Some(Node::Stmt(stmt)),
        Expr::parse(input, ctx) => Some(expr) => return Some(Node::Expr(expr))
    }

    None
}

/// Code Block (List of Statements)
/// ===========
///
/// # Syntax:
/// `{ ... }`
///
/// # Parsing
/// See [parse_block] function.
type Block = (Vec<Node>, Context);

pub fn parse_block(
    input: &mut Peekable<impl Iterator<Item = Token>>,
    ctx: &mut Context,
    is_param: bool,
) -> Option<Block> {
    if is_param {
        input.next()?;
    }

    _ = expect_token(input, Token::LeftBrace)?;

    let mut stmts = Vec::<Node>::new();

    let mut braces = 1;

    let mut new_ctx = ctx.clone();

    loop {
        if braces <= 0 {
            break;
        }

        let token = input.peek()?;

        match token {
            Token::LeftBrace => braces += 1,
            Token::RightBrace => braces -= 1,
            _ => match parse_node(input, &mut new_ctx) {
                Some(stmt) => {
                    stmts.push(stmt);
                    continue;
                }
                None => error_and_return!(
                    #[Info]
                    "Failed to parse statement."
                ),
            },
        };

        input.next();
    }

    Some((stmts, new_ctx))
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
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
