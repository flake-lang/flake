#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer,
    exclusive_range_pattern,
    slice_pattern
)]

use std::collections::HashMap;

use lexer::create_lexer;
use parser::TokenStream;

extern crate inkwell;

mod ast;
mod compile;
mod lexer;
mod parser;
mod token;

#[cfg(test)]
mod tests;

fn main() {
    let code = include_str!("../test.fl");
    let mut context = ast::Context {
        locals: HashMap::new(),
        can_return: false,
        types: [].into(),
    };

    let mut tokens = create_lexer(code);
    let mut tokens_peekable = tokens.peekable();

    let mut statements = Vec::<ast::Statement>::new();

    //  dbg!(ast::Statement::parse(&mut tokens_peekable));

    dbg!(&code);

    loop {
        if tokens_peekable.peek() == Some(&token::Token::EOF) {
            break;
        }

        if let Some(ast_node) = ast::Statement::parse(&mut tokens_peekable, &mut context) {
            // println!("{:#?}", &ast_node);
            //  println!("TYPE = {:#?}", ast::infer_expr_type(ast_node.clone()));
            statements.push(ast_node);
        }
    }

    println!("{:?}", statements);
    println!("{:?}", context);

    // let ast = dbg!(parser::parse_node(&mut tokens_peekable)).expect("Failed to parse syntax tree");

    let mut compiler = compile::Compiler::new();

    // compiler.compile_with_prelude_main(vec![ast]);
}
