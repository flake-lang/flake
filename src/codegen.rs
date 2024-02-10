//! Compiling

use core::{ascii, slice::SlicePattern};
use std::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    collections::{btree_map::ValuesMut, HashMap},
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::Deref,
    sync::{atomic::compiler_fence, Arc},
};

use colored::Colorize;
use inkwell::{
    attributes::Attribute,
    basic_block::{self, BasicBlock},
    builder::{Builder, BuilderError},
    context::Context as LLVMContext,
    module::Module,
    types::{
        AnyType, AnyTypeEnum, ArrayType, AsTypeRef, BasicMetadataTypeEnum, BasicType,
        BasicTypeEnum, IntType, PointerType,
    },
    values::{
        AnyValue, AnyValueEnum, BasicMetadataValueEnum, BasicValue, BasicValueEnum, BasicValueUse,
        FunctionValue, InstructionValue, IntValue, PointerValue,
    },
    AddressSpace,
};
use itertools::Either;
use serde::Serialize;
use serde_json::ser::CharEscape;

use crate::{
    ast::{
        self, infer_expr_type, Block, BuiltinMarkerFunc, Context as ASTContext, Expr, Function,
        FunctionSignature, Statement, Type, Value,
    },
    cast, compile,
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
    pub target: Box<dyn Any>,
    pub eval_context: &'a mut crate::eval::Context,
}

impl<'a> Context<'a> {
    #[inline]
    pub fn fn_value(&self) -> FunctionValue<'a> {
        self.fn_value_opt.expect("function value to present")
    }
}

#[derive(Debug)]
pub enum CompilerError<'a> {
    LLVMBuilder(BuilderError),
    ComptimeEval(&'a str),
    VariableNotFound(String),
    Internal(&'a str),
    Other(&'a str),
    OtherString(String),
}

pub fn discard_value<T>(_: T) {}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    #[inline]
    pub fn get_function(&self, name: &'_ str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    pub fn create_uninitalized_variable<const REG: bool>(
        &mut self,
        ty: Type,
        name: String,
    ) -> &mut Self {
        let alloca =
            create_entry_block_alloca(self, name.as_str(), basic_llvm_type(self.llvm_context, ty));

        if REG {
            self.context.variables.insert(name, alloca);
        }
        self
    }

    pub fn compile_body(
        &mut self,
        func: FunctionValue<'ctx>,
        block: Block,
    ) -> Result<(), CompilerError> {
        let body = self.llvm_context.append_basic_block(func, "body");

        self.builder.build_unconditional_branch(body);

        self.builder.position_at_end(body);

        let statements = block.0;

        for stmt in statements {
            self.compile_stmt(stmt).unwrap();
        }

        Ok(())
    }

    #[inline]
    pub fn build_ret(&mut self, expr: Expr) -> Result<InstructionValue<'ctx>, CompilerError> {
        let typeof_expr = infer_expr_type(expr.clone());

        if typeof_expr == Type::Void {
            return self
                .builder
                .build_return(None)
                .map_err(|err| CompilerError::LLVMBuilder(err));
        } else {
            let value = self.compile_expr(expr).unwrap();
            return self
                .builder
                .build_return(Some(&value))
                .map_err(|err| CompilerError::LLVMBuilder(err));
        }
    }

    pub fn compile_stmt(
        &mut self,
        stmt: Statement,
    ) -> Result<InstructionValue<'ctx>, CompilerError> {
        match stmt {
            Statement::Return { value } => self.build_ret(value),
            Statement::Let { ty, ident, value } => {
                self.create_uninitalized_variable::<true>(ty, ident.clone());

                let compiled_value = self.compile_expr(value).unwrap();

                self.store_in_variable(ident, compiled_value)
            }
            Statement::FunctionCall {
                func,
                args,
                intrinsic,
            } => self.compile_function_call(func, args, intrinsic)?.either(
                |v| {
                    v.as_instruction_value()
                        .ok_or(CompilerError::Other("invalid instruction value"))
                },
                |i| Ok(i),
            ),
            _ => todo!(),
        }
    }

    #[inline]
    pub fn store_in_variable(
        &mut self,
        name: String,
        value: impl BasicValue<'ctx>,
    ) -> Result<InstructionValue<'ctx>, CompilerError> {
        let var_ptr = self
            .context
            .variables
            .get(&name)
            .ok_or(CompilerError::VariableNotFound(name))?;

        self.builder
            .build_store(*var_ptr, value)
            .map_err(|err| CompilerError::LLVMBuilder(err))
    }

    pub fn compile_func(&mut self) -> Result<FunctionValue<'ctx>, CompilerError> {
        self.context.variables.clear();

        let func = self
            .target
            .downcast_ref::<Function>()
            .ok_or(CompilerError::Internal("target isn't a function"))?;
        let sig = func.sig.clone();

        let fn_value = self
            .compile_sig(func.sig.clone())
            .map_err(|err| CompilerError::Other("err"))?;

        if func.body.is_none() {
            return Ok(fn_value);
        }

        let fn_prelude = self.llvm_context.append_basic_block(fn_value, "_prelude");

        self.builder.position_at_end(fn_prelude);

        self.context.fn_value_opt = Some(fn_value);

        for (i, arg) in fn_value.get_param_iter().enumerate() {
            let arg_decl = sig.args[i].clone();

            let alloca = create_entry_block_alloca(
                self,
                arg_decl.0.clone().as_str(),
                basic_llvm_type(self.llvm_context, arg_decl.1),
            );

            self.builder.build_store(alloca, arg).unwrap();

            self.context.variables.insert(arg_decl.0, alloca);
        }

        let body = self
            .compile_body(fn_value, func.body.clone().unwrap())
            .unwrap();

        Ok(fn_value)
    }

    pub fn compile_sig(&self, sig: FunctionSignature) -> Result<FunctionValue<'ctx>, &'_ str> {
        let argc = sig.args.len();

        let args = sig
            .args
            .iter()
            .map(|arg| generic_llvm_type_inlined!(self.llvm_context, arg.1.clone()))
            .collect::<Vec<BasicMetadataTypeEnum>>();

        let fn_type = if sig.return_type == Type::Void {
            self.llvm_context
                .void_type()
                .fn_type(args.as_slice(), false)
        } else {
            basic_llvm_type(self.llvm_context, sig.return_type).fn_type(args.as_slice(), false)
        };

        let function = self.module.add_function(
            sig.name
                .unwrap_or(format!("_Un{}", fn_type.as_type_ref() as usize))
                .as_str(),
            fn_type,
            None,
        );

        for (i, arg) in function.get_param_iter().enumerate() {
            arg.set_name(sig.args[i].0.as_str())
        }

        Ok(function)
    }

    pub fn compile_expr(&mut self, expr: Expr) -> Result<BasicValueEnum<'ctx>, CompilerError> {
        match expr {
            Expr::Constant(val) => Ok(llvm_value(self.llvm_context, val)),
            Expr::Comptime(boxed) => Ok(llvm_value(
                self.llvm_context,
                eval_expr(*boxed, self.eval_context)
                    .ok_or(CompilerError::ComptimeEval("failed to parse expression"))?,
            )
            .into()),
            Expr::Variable { ident: name, .. } => {
                let ptr = self
                    .context
                    .variables
                    .get(&name)
                    .ok_or(CompilerError::VariableNotFound(name.clone()))?;

                return Ok(self
                    .builder
                    .build_load(*ptr, name.clone().as_str())
                    .map_err(|err| CompilerError::LLVMBuilder(err))?
                    .into());
            },
            Expr::Cast { into, expr } => self.compile_expr(*expr),
            Expr::FunctionCall {
                func,
                args,
                intrinsic,
                ..
            } => self
                .compile_function_call(func, args, intrinsic)
                .map(|either| {
                    either.left().ok_or(CompilerError::Other(
                        "cannot call void function is expression",
                    ))
                })?,
            _ => todo!(),
        }
    }

    pub fn invoke_intrinsic(
        &mut self,
        name: String,
        args: Vec<Expr>,
    ) -> Result<Either<BasicValueEnum<'ctx>, InstructionValue<'ctx>>, CompilerError> {
        Ok(match name.as_str() {
            "unreachable" => Either::Right(
                self.builder
                    .build_unreachable()
                    .map_err(|err| CompilerError::LLVMBuilder(err))?,
            ),
            "__goto_label" => {
                let fn_value = self.context.fn_value();
                let arg0 = eval_expr(args.get(0).unwrap().clone(), self.eval_context).unwrap();
                let name = cast!(arg0.deref(), TokenKind::String, str);

                let block = fn_value
                    .get_basic_block_iter()
                    .find(|b| b.get_name().to_str() == Ok(name.as_str()))
                    .expect("invalid block jump");

                Either::Right(
                    self.builder
                        .build_unconditional_branch(block)
                        .map_err(|err| CompilerError::LLVMBuilder(err))?,
                )
            }
            _ => panic!("invalid intrinsic {:?}!", name),
        })
    }

    pub fn compile_function_call(
        &mut self,
        name: String,
        args: Vec<Expr>,
        intrinsic: bool,
    ) -> Result<Either<BasicValueEnum<'ctx>, InstructionValue<'ctx>>, CompilerError> {
        if intrinsic {
            return Ok(self.invoke_intrinsic(name.clone(), args.clone())?);
        }

        let func_value = self
            .module
            .get_function(name.as_str())
            .ok_or(CompilerError::Other("function not found"))?;

        let llvm_args = args
            .iter()
            .map(|v| llvm_value_to_metavalue(self.compile_expr(v.clone()).unwrap()))
            .collect::<Vec<BasicMetadataValueEnum<'ctx>>>();

        self.builder
            .build_call(func_value, llvm_args.as_slice(), "")
            .map_err(|err| CompilerError::LLVMBuilder(err))
            .map(|site| site.try_as_basic_value())
    }
}

#[inline(always)]
pub fn llvm_value_to_metavalue<'ctx>(v: BasicValueEnum<'ctx>) -> BasicMetadataValueEnum<'ctx> {
    match v {
        BasicValueEnum::ArrayValue(a) => BasicMetadataValueEnum::ArrayValue(a),
        BasicValueEnum::IntValue(x) => BasicMetadataValueEnum::IntValue(x),
        BasicValueEnum::FloatValue(f) => BasicMetadataValueEnum::FloatValue(f),
        BasicValueEnum::PointerValue(p) => BasicMetadataValueEnum::PointerValue(p),
        BasicValueEnum::StructValue(s) => BasicMetadataValueEnum::StructValue(s),
        _ => unimplemented!(),
    }
}

/// Creates a new stack allocation instruction in the entry block of the function.
fn create_entry_block_alloca<'a, 'ctx>(
    compiler: &Compiler<'a, 'ctx>,
    name: &str,
    ty: BasicTypeEnum<'ctx>,
) -> PointerValue<'ctx> {
    let builder = compiler.llvm_context.create_builder();

    let entry = compiler.context.fn_value().get_last_basic_block().unwrap();

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
        Type::Char => ctx.i8_type().as_basic_type_enum(),
        Type::String => ctx.i8_type().ptr_type(AddressSpace::default()).into(),
        Type::Void => ctx.const_struct(&[], false).get_type().as_basic_type_enum(),
        Type::Pointee { target_ty } => basic_llvm_type(ctx, *target_ty)
            .ptr_type(AddressSpace::default())
            .as_basic_type_enum(),
        _ => panic!(
            "The type {:?} is not a valid llvm basic type",
            ast_type.to_string()
        ),
    }
}

macro generic_llvm_type_inlined($ctx:expr, $t:expr) {
    match $t {
        Type::Boolean => return $ctx.bool_type().into(),
        Type::UnsignedInt { bits } => $ctx.custom_width_int_type(bits as u32).into(),
        Type::Int { bits } => $ctx.custom_width_int_type(bits as u32 as u32).into(),
        Type::Float { bits: 32 } => $ctx.f32_type().into(),
        Type::Char => $ctx.i8_type().into(),
        Type::String => $ctx.i8_type().ptr_type(AddressSpace::default()).into(),
        Type::Pointee { target_ty } => basic_llvm_type($ctx, *target_ty)
            .ptr_type(AddressSpace::default())
            .into(),
        _ => panic!("The type {:?} is not a valid llvm type!", $t.to_string()),
    }
}

pub fn llvm_type<'ctx>(ctx: &'ctx LLVMContext, ast_type: Type) -> AnyTypeEnum<'ctx> {
    generic_llvm_type_inlined!(ctx, ast_type)
}

macro generic_llvm_value_inlined($ctx:expr, $v:expr) {
    match ($v.deref(), $v.get_type()) {
        (TokenKind::Boolean(b), _) => $ctx
            .bool_type()
            .const_int(boolean_to_uint(*b), false)
            .into(),
        (TokenKind::Number(n), t) => const_int($ctx, t, *n as u64).into(),
        (TokenKind::String(s), _) => $ctx.const_string(s.as_bytes(), false).into(),
        (TokenKind::Char(c), t) => $ctx.i8_type().const_int(*c as u64, false).into(),
        (_, t) => panic!("invalid value type {:?}", t.to_string()),
    }
}

#[inline]
pub fn llvm_value<'ctx>(ctx: &'ctx LLVMContext, v: Value) -> BasicValueEnum<'ctx> {
    generic_llvm_value_inlined!(ctx, v)
}

#[inline]
pub fn llvm_value_instance<'ctx, 'a>(
    ctx: &'ctx LLVMContext,
    v: Value,
) -> Option<&'a dyn BasicValue<'ctx>> {
    match v.get_type() {
        Type::Void => None,
        Type::Never => panic!("an instance of the type \"never\" cannot be created"),
        _ => todo!(),
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
