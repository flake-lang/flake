//! Compiling

use core::ascii;
use std::{
    collections::{btree_map::ValuesMut, HashMap},
    hash::Hash,
    ops::Deref,
};

use inkwell::{
    basic_block,
    builder::Builder,
    context::Context as LLVMContext,
    module::Module,
    types::{AnyType, AnyTypeEnum, AsTypeRef, BasicType, BasicTypeEnum, IntType, PointerType},
    values::{
        AnyValue, AnyValueEnum, BasicValue, BasicValueEnum, BasicValueUse, FunctionValue, IntValue,
        PointerValue,
    },
    AddressSpace,
};
use serde::Serialize;
use serde_json::ser::CharEscape;

use crate::{
    ast::{Expr, Function, FunctionSignature, Type, Value},
    eval::{self, eval_expr},
    token::TokenKind,
};

#[derive(Debug)]
pub struct Context<'a> {
    pub variables: HashMap<String, PointerValue<'a>>,
    pub fn_value_opt: Option<FunctionValue<'a>>,
}

#[derive(Debug)]
pub struct Compiler<'a, 'ctx> {
    pub llvm_context: &'ctx LLVMContext,
    pub context: Context<'ctx>,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,
    pub function: &'a Function,
    pub eval_context: &'a mut crate::eval::Context,
}

impl<'a> Context<'a> {
    #[inline]
    pub fn fn_value(&self) -> FunctionValue<'a> {
        self.fn_value_opt.expect("function value to present")
    }
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    #[inline]
    pub fn get_function(&self, name: &'_ str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    pub fn create_uninitalized_variable(&mut self, ty: Type, name: String) {
        let alloca =
            create_entry_block_alloca(self, name.as_str(), basic_llvm_type(self.llvm_context, ty));

        self.context.variables.insert(name, alloca);
    }

    pub fn compile_sig(&self, sig: FunctionSignature) -> Result<FunctionValue<'ctx>, &'_ str> {
        let fn_type = if sig.return_type == Type::Void {
            self.llvm_context.void_type().fn_type(&[], false)
        } else {
            basic_llvm_type(self.llvm_context, sig.return_type).fn_type(&[], false)
        };

        let function = self.module.add_function(
            sig.name
                .unwrap_or(format!("_Unnammed{}", fn_type.as_type_ref() as usize))
                .as_str(),
            fn_type,
            None,
        );

        let entry = self.llvm_context.append_basic_block(function, "entry");

        self.builder.position_at_end(entry);

        Ok(function)
    }

    pub fn compile_expr(&mut self, expr: Expr) -> Result<AnyValueEnum<'ctx>, &'_ str> {
        match expr {
            Expr::Constant(val) => Ok(llvm_value(self.llvm_context, val).into()),
            Expr::Comptime(boxed) => Ok(llvm_value(
                self.llvm_context,
                eval_expr(*boxed, self.eval_context)
                    .ok_or("failed to eval compiletime expression")?,
            )
            .into()),
            Expr::Variable { ident: name, .. } => {
                let ptr = self
                    .context
                    .variables
                    .get(&name)
                    .ok_or("cannot find variable.")?;

                return Ok(self
                    .builder
                    .build_load(*ptr, name.as_str())
                    .map_err(|ref err| "failed to load value from pointer.")?
                    .into());
            }
            _ => todo!(),
        }
    }
}

/// Creates a new stack allocation instruction in the entry block of the function.
fn create_entry_block_alloca<'a, 'ctx>(
    compiler: &Compiler<'a, 'ctx>,
    name: &str,
    ty: BasicTypeEnum<'ctx>,
) -> PointerValue<'ctx> {
    let builder = compiler.llvm_context.create_builder();

    let entry = compiler.context.fn_value().get_first_basic_block().unwrap();

    match entry.get_first_instruction() {
        Some(first_instr) => builder.position_before(&first_instr),
        None => builder.position_at_end(entry),
    }

    builder.build_alloca(ty, name).unwrap()
}

pub fn basic_llvm_type<'ctx>(ctx: &'ctx LLVMContext, ast_type: Type) -> BasicTypeEnum<'ctx> {
    match ast_type {
        Type::Boolean => return ctx.bool_type().as_basic_type_enum(),
        Type::UnsignedInt { bits } => ctx.custom_width_int_type(bits as u32).as_basic_type_enum(),
        Type::Int { bits } => ctx
            .custom_width_int_type(bits as u32 as u32)
            .as_basic_type_enum(),
        Type::Float { bits: 32 } => ctx.f32_type().as_basic_type_enum(),
        Type::String => ctx.i8_type().ptr_type(AddressSpace::default()).into(),
        Type::Pointee { target_ty } => basic_llvm_type(ctx, *target_ty)
            .ptr_type(AddressSpace::default())
            .as_basic_type_enum(),
        _ => panic!(
            "The type {:?} is not a valid llvm basic type",
            ast_type.to_string()
        ),
    }
}

pub fn llvm_type<'ctx>(ctx: &'ctx LLVMContext, ast_type: Type) -> AnyTypeEnum<'ctx> {
    match ast_type {
        Type::Void => ctx.void_type().into(),
        Type::Function(boxed) => llvm_type(ctx, boxed.return_type)
            .into_pointer_type()
            .fn_type(&[], false)
            .into(),
        t => basic_llvm_type(ctx, t).as_any_type_enum(),
    }
}

pub fn llvm_value<'ctx>(ctx: &'ctx LLVMContext, v: Value) -> BasicValueEnum<'ctx> {
    match (v.deref(), v.get_type()) {
        (TokenKind::Boolean(b), _) => ctx.bool_type().const_int(boolean_to_uint(*b), false).into(),
        (TokenKind::Number(n), t) => const_int(ctx, t, *n as u64).into(),
        (TokenKind::String(s), _) => ctx.const_string(s.as_bytes(), false).into(),
        (_, t) => panic!("invalid value type {:?}", t.to_string()),
    }
}

#[inline]
pub fn const_int<'ctx>(ctx: &'ctx LLVMContext, ty: Type, v: u64) -> IntValue<'ctx> {
    match ty {
        Type::UnsignedInt { bits } => ctx.custom_width_int_type(bits as u32).const_int(v, false),
        Type::Int { bits } => ctx.custom_width_int_type(bits as u32).const_int(v, true),
        _ => panic!("{:?} isn't a valid interger type", ty.to_string()),
    }
}

#[inline(always)]
pub const fn boolean_to_uint(v: bool) -> u64 {
    match v {
        true => 1,
        false => 0,
    }
}
