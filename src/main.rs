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

    for token in tokens {
        match token {
            token::Token::Identifier(_) => continue,
            _ => {}
        }
        println!("==> Found Token: {:?}", token)
    }

    // let ast = dbg!(parser::parse_node(&mut tokens_peekable)).expect("Failed to parse syntax tree");

    let mut compiler = compile::Compiler::new();

    // compiler.compile_with_prelude_main(vec![ast]);
}
