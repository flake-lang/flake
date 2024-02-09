#![allow(warnings)]
#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer,
    exclusive_range_pattern,
    slice_pattern,
    proc_macro_hygiene,
    stmt_expr_attributes,
    decl_macro
)]

#[macro_use]
extern crate macros;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::exit,
};

use inkwell::{passes::PassManagerSubType, values::FunctionValue};
use itertools::Itertools;
use lexer::create_lexer;
use parser::TokenStream;

use crate::{
    ast::{parse_node, FnSig, Function, MarkerImpl, Type},
    eval::{eval_expr, Context as EvalContext},
};

extern crate inkwell;

pub mod ast;
pub mod builtins;
mod codegen;
mod compile;
mod eval;
mod feature;
mod lexer;
mod parser;
mod pipeline;
mod token;

#[cfg(test)]
mod tests;

fn main() {
    let code = include_str!("../test.fl");
    let mut context = ast::Context {
        locals: HashMap::new(),
        can_return: true,
        types: [].into(),
        feature_gates: {
            use feature::FeatureKind::*;

            HashMap::from(include!("../features.specs"))
        },
        markers: HashMap::from_iter(
            builtins::MARKERS
                .iter()
                .map(|(n, f)| ((*n).to_owned(), ast::MarkerImpl::BuiltIn(*f))),
        ),
    };

    context.register_local_variable("version".to_owned(), Type::String);

    let mut eval_context = EvalContext {
        variables: HashMap::new(),
    };

    let mut dbg_trokens = create_lexer(code.clone()).collect_vec();

    dbg!(dbg_trokens);

    let mut tokens = create_lexer(code);

    let mut tokens_peekable = tokens.peekable();

    let mut statements = Vec::<ast::Expr>::new();

    loop {
        if tokens_peekable.peek() == Some(&token::Token::EOF) {
            break;
        }

        if let Some(ast_node) = ast::Expr::parse(&mut tokens_peekable, &mut context) {
            // eval::eval_statement(ast_node.clone(), &mut eval_context);
            //   println!("{:#?}", &ast_node);
            //  println!("TYPE = {:#?}", ast::infer_expr_type(ast_node.clone()));
            statements.push(dbg!(ast_node));
        }
    }

    if pipeline::COMPILER_PIPELINE
        .read()
        .unwrap()
        .needs_terminate()
    {
        eprintln!(
            "Failed to compile program due to {} errors!",
            pipeline::COMPILER_PIPELINE
                .read()
                .unwrap()
                .errors
                .load(std::sync::atomic::Ordering::SeqCst)
        );
        exit(1);
    }

    println!("{:#?}", statements);
    println!("{:#?}", context);

    let serilaized = format!(
        "{}",
        serde_json::to_string(&serde_json::json! ({
            "tree": statements,
            "context": context,
            "is_transparent": false
        }))
        .unwrap()
    );

    std::fs::write("test.fl.json", serilaized);

    // let ast = dbg!(parser::parse_node(&mut tokens_peekable)).expect("Failed to parse syntax tree");

    // let mut compiler = compile::Compiler::new();

    // compiler.compile_with_prelude_main(vec![ast]);

    let llvm_context = inkwell::context::Context::create();
    let module = llvm_context.create_module("test");
    let builder = llvm_context.create_builder();

    let mut compiler = codegen::Compiler {
        llvm_context: &llvm_context,
        context: codegen::Context {
            variables: HashMap::new(),
            fn_value_opt: None,
        },
        builder: &builder,
        module: &module,
        function: &Function {
            name: "main".to_owned(),
            sig: ast::FnSig {
                args: vec![],
                return_type: Type::Boolean,
                name: Some("main".to_owned()),
            },
            body: (vec![], context.clone()),
        },
        eval_context: &mut eval_context,
    };

    compiler.context.fn_value_opt = Some(
        compiler
            .compile_sig(FnSig {
                args: vec![],
                name: Some("main".to_owned()),
                return_type: Type::Void,
            })
            .unwrap(),
    );

    compiler.create_uninitalized_variable(Type::String, "version".to_owned());

    for expr in statements {
        dbg!(compiler.compile_expr(expr));
    }

    compiler.module.print_to_stderr();
    compiler
        .module
        .write_bitcode_to_path(Path::new("../test.fl.bc"));

    dbg!(eval_context);
}
