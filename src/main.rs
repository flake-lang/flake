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

use std::{collections::HashMap, process::exit};

use lexer::create_lexer;
use parser::TokenStream;

use crate::ast::MarkerImpl;

extern crate inkwell;

pub mod ast;
pub mod builtins;
mod codegen;
mod compile;
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

    let mut tokens = create_lexer(code);
    let mut tokens_peekable = tokens.peekable();

    let mut statements = Vec::<ast::Node>::new();

    //  dbg!(ast::Statement::parse(&mut tokens_peekable));

    /*  pipeline::COMPILER_PIPELINE
    .read()
    .unwrap()
    .process_message(pipeline::Message::Error {
        err: "test".to_owned(),
        notes: vec!["abc".to_owned(), "123".to_owned()],
    });*/

    /*  pipeline_send!(
            #[Error]
            "This is an error!",
            "You can write more about it here...",
            "even in multiple lines!"
        );
    */
    pipeline_send!(
        #[Warning]
        "This is an warning!",
        "You can write more about it here...",
        "even in multiple lines!"
    );

    pipeline_send!(
        #[Info]
        "This is an information!",
        "You can write more about it here...",
        "even in multiple lines!"
    );
    /*  pipeline_send!(
        #[_Unimplemented]
        "unimplemented!!",
        "... will panic!"
    ); */

    loop {
        if tokens_peekable.peek() == Some(&token::Token::EOF) {
            break;
        }

        if let Some(ast_node) = ast::parse_node(&mut tokens_peekable, &mut context) {
            //   println!("{:#?}", &ast_node);
            //  println!("TYPE = {:#?}", ast::infer_expr_type(ast_node.clone()));
            statements.push(ast_node);
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
}
