#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer,
    exclusive_range_pattern,
    slice_pattern
)]

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

    let mut tokens = create_lexer(code);
    let mut tokens_peekable = tokens.peekable();

    let mut exprs = Vec::<ast::Expr>::new();

    loop {
        if tokens_peekable.peek() == Some(&token::Token::Semicolon) {
            tokens_peekable.next();
        }

        if let Some(ast_node) = ast::Expr::parse(&mut tokens_peekable) {
            println!("{:#?}", &ast_node);
            println!("TYPE = {:#?}", ast::infer_expr_type(ast_node.clone()));
            exprs.push(ast_node);
        } else {
            break;
        }
    }

    println!("{:#?}", exprs);

    // let ast = dbg!(parser::parse_node(&mut tokens_peekable)).expect("Failed to parse syntax tree");

    let mut compiler = compile::Compiler::new();

    // compiler.compile_with_prelude_main(vec![ast]);
}
