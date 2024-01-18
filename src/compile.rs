//! Flake Compiler (using LLVM)

use std::path::Path;

use inkwell::{
    self,
    builder::Builder,
    context::Context,
    module::Module,
    types,
    types::IntType,
    values::{AnyValue, IntValue},
    *,
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
            let recursive_builder = RecursiveBuilder::new(i64_type, &builder);
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
}

impl<'a> RecursiveBuilder<'a> {
    pub fn new(i64_type: IntType<'a>, builder: &'a Builder) -> Self {
        Self { i64_type, builder }
    }
    pub fn build(&self, ast: &Node) -> IntValue {
        match ast {
            Node::Int(n) => self.i64_type.const_int(*n as u64, true),
            Node::UnaryExpr { op, child } => {
                let child = self.build(child);
                match op {
                    Operator::Minus => child.const_neg(),
                    Operator::Plus => child,
                    _ => child,
                }
            }
            Node::BinaryExpr { op, lhs, rhs } => {
                let left = self.build(lhs);
                let right = self.build(rhs);

                match op {
                    Operator::Plus => self
                        .builder
                        .build_int_add(left, right, "plus_temp"),
                    Operator::Minus => self
                        .builder
                        .build_int_sub(left, right, "minus_temp"),
                    _ => unimplemented!("Pipeline Error!\nC012: LLVM RecursiveBuilder<'_> encoutered a unknown operator in binary expression!")
                }
            }
        }
    }
}
