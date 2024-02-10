#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer,
    exclusive_range_pattern,
    slice_pattern,
    proc_macro_hygiene,
    extend_one,
    stmt_expr_attributes,
    exit_status_error,
    decl_macro
)]
#![allow(warnings)]

#[macro_use]
extern crate macros;

use std::{
    collections::HashMap,
    fs,
    ops::IndexMut,
    path::{Path, PathBuf},
    process::exit,
    sync::atomic::compiler_fence,
};

use clap::Parser;
use clap_derive::Parser;
use colored::Colorize;
use inkwell::{passes::PassManagerSubType, targets::TargetTriple, values::FunctionValue};
use itertools::Itertools;
use lexer::create_lexer;
use parser::TokenStream;

use crate::{
    ast::{parse_node, FnSig, Function, MarkerImpl, Type},
    codegen::Compiler,
    eval::{eval_expr, Context as EvalContext},
    token::TokenKind,
};

extern crate inkwell;

pub mod ast;
pub mod builtins;
mod codegen;
mod compile;
mod eval;
mod feature;
mod intrinsics;
mod lexer;
mod parser;
mod pipeline;
mod token;

#[cfg(test)]
mod tests;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    pub file: String,

    #[arg(short, long)]
    pub output: Option<String>,
    #[arg(long)]
    pub target: Option<String>,

    #[arg(long)]
    pub display_llvm_ir: Option<bool>,
}

fn main() {
    let cli = Cli::parse();

    let code = fs::read_to_string(cli.file.clone()).unwrap();
    let output_str = format!("{}.bc", cli.file);

    run(code.as_str(), output_str.clone().into());

    let obj_file = format!("{}.o", cli.file);

    std::process::Command::new("llc-14")
        .arg("--filetype=obj")
        .args(["-o", obj_file.as_str()])
        .arg(output_str)
        .status()
        .expect("failed to invoke llvm")
        .exit_ok()
        .expect(format!("{}", "Failed to compile object file".red()).as_str());

    std::process::Command::new("cc")
        .args(["-o", cli.output.clone().unwrap().as_str()])
        .arg(obj_file)
        .arg("-static")
        .status()
        .expect("failed to launch c-compiler.")
        .exit_ok()
        .expect(format!("{}", "Failed to link using the c-compiler".red()).as_str());

    println!(
        "{}, use `{}` to run.",
        "Done".green(),
        cli.output.unwrap().italic()
    );
}

fn run(code: &str, output: PathBuf) {
    let mut context = ast::Context {
        locals: HashMap::new(),
        can_return: true,
        functions: HashMap::new(),
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

    let mut tokens = create_lexer(code);

    let mut tokens_peekable = tokens.peekable();

    let mut statements = Vec::<ast::Node>::new();

    loop {
        if tokens_peekable.peek() == None {
            break;
        }

        if let Some(ast_node) = ast::parse_node(&mut tokens_peekable, &mut context) {
            // eval::eval_statement(ast_node.clone(), &mut eval_context);
            //   println!("{:#?}", &ast_node);
            //  println!("TYPE = {:#?}", ast::infer_expr_type(ast_node.clone()));
            statements.push(ast_node);
        } else {
            break;
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
        target: Box::new(()),
        eval_context: &mut eval_context,
    };

    //  compiler.compile_func();

    //   compiler.create_uninitalized_variable(Type::String, "version".to_owned());

    for expr in statements {
        match expr {
            ast::Node::Item(ast::Item::Function(f)) => {
                compiler.target = Box::new(f);
                compiler.compile_func();
            }
            _ => todo!(),
        }
    }

    if let Err(llvm_str) = compiler.module.verify() {
        eprintln!(
            "=== LLVM ERROR ===\n{}{}\n",
            llvm_str.to_string().replace("\n", "\n ==> ").red().bold(),
            "[Generated IR may be invalid!]".red().bold()
        );
    }

    compiler.module.write_bitcode_to_path(output.as_path());
}
