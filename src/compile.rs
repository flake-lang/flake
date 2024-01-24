//! Flake Compiler (using LLVM)

use std::{ops::Deref, path::Path};

/*

use inkwell::{
    self,
    builder::Builder,
    context::Context,
    module::Module,
    types,
    types::{AnyType, AnyTypeEnum, IntType, BasicType, BasicTypeEnum},
    values::{AnyValue, AnyValueEnum, IntMathValue, IntValue},
    *, basic_block::BasicBlock,
};

use crate::ast::{Node, Operator};

pub struct Compiler {
    llvm_context: Context,
    builtin_modules: Vec<()>,
    compiler_version: String,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            llvm_context: Context::create(),
            builtin_modules: vec![],
            compiler_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    pub fn compile_with_prelude_main(&mut self, ast: Vec<crate::ast::Node>) {
        let module = self.llvm_context.create_module("__flakec_test");

        let builder = self.llvm_context.create_builder();

        let i64_type = self.llvm_context.i64_type();
        let fn_type = i64_type.fn_type(&[], false);

        let function = module.add_function("main", fn_type, None);
        let basic_block = self.llvm_context.append_basic_block(function, "entry");

        builder.position_at_end(basic_block);

        for node in ast {
            let recursive_builder = RecursiveBuilder::new(i64_type, &builder, &self.llvm_context);
            let return_value = recursive_builder.build(&node);
            let _ = builder.build_return(Some(&return_value));
        }
        println!(
            "Generated LLVM IR: {}",
            function.print_to_string().to_string()
        );

        module.set_source_file_name("__flakec_builtin.test");

        if let Err(err) = module.verify() {
            eprintln!("LLVM Output: \n{}", err.to_string());
        } else {
            module.write_bitcode_to_path(Path::new("test.fl.bc"));
        }
    }
}

pub struct RecursiveBuilder<'a> {
    i64_type: IntType<'a>,
    builder: &'a Builder<'a>,
    context: &'a Context,
}

use crate::ast;

impl<'a> RecursiveBuilder<'a> {
    pub fn new(i64_type: IntType<'a>, builder: &'a Builder, ctx: &'a Context) -> Self {
        Self {
            i64_type,
            builder,
            context: ctx,
        }
    }

    pub fn build_binary_expr(&self, op: Operator, rhs: ast::Expr, lhs: ast::Expr){
        match (op, ast::infer_expr_type(rhs)){
            (Operator::Plus, ast::Type::UnsignedInt { .. }) => self.builder.build_int_nsw_neg(, )
        }
    }

    pub fn build_expr(&self, ast: ast::Expr) {
        match ast{
            ast::Expr::Constant(v) => self.build_value(v),
            ast::Expr::Binary { op, rhs, lhs } =>
        }
    }

    pub fn build_type(&self, ast_ty: crate::ast::Type) -> Result<AnyTypeEnum, String> {
        match ast_ty {
            ast::Type::Boolean => Ok(self.context.bool_type().as_any_type_enum()),
            ast::Type::Void => Ok(self.context.void_type().as_any_type_enum()),
            _ => panic!("compiler error: invalid type: {:?}", ast_ty),
        }
    }


    pub fn build_basic_type(&self, ast_ty: crate::ast::Type) -> Result<BasicTypeEnum, String> {
        match ast_ty {
            ast::Type::Boolean => Ok(self.context.bool_type().as_basic_type_enum()),
            ast::Type::UnsignedInt { bits: 32 } => Ok(self.context.i32_type().as_basic_type_enum()),
            _ => panic!("compiler error: invalid type: {:?}", ast_ty),
        }
    }


}
*/
