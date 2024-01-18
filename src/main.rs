#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer
)]
#![feature(exclusive_range_pattern)]

use lexer::create_lexer;

mod lexer;
mod token;
mod ast;

#[cfg(test)]
mod tests;

fn main() {
    let code = "2 * (5 + 10)";

  let mut tokens = create_lexer(code).peekable();   

    dbg!(ast::parse_node(&mut tokens));
}
