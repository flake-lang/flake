//! LLVM Codegen

use inkwell::debug_info::LLVMDWARFTypeEncoding;

use crate::ast::{self, *};

pub trait Codegen {
    type Output;
    type Context;

    fn build(&self, ctx: &mut Self::Context) -> Result<Self::Output, String>;
}
