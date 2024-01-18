#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer,
    exclusive_range_pattern,
    slice_pattern
)]

use lexer::create_lexer;

extern crate inkwell;

mod ast;
mod compile;
mod lexer;
mod token;

#[cfg(test)]
mod tests;

fn main() {
    let code = include_str!("../test.fl");

    let mut tokens = create_lexer(code).peekable();

    let ast = dbg!(ast::parse_node(&mut tokens)).expect("Failed to parse syntax tree");

    let mut compiler = compile::Compiler::new();

    compiler.compile_with_prelude_main(vec![ast]);
}
