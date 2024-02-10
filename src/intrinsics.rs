use std::{any::Any, collections::HashMap};

use inkwell::{
    context::Context,
    types::{AnyTypeEnum, BasicMetadataTypeEnum, BasicTypeEnum},
    values::BasicValueEnum,
};

use crate::{ast::Expr, codegen::Compiler};

pub type Intrinsic<'ctx> = fn(&'ctx Context, Vec<Expr>) -> Result<BasicValueEnum<'ctx>, &str>;
