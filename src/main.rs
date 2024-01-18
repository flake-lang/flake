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

#[cfg(test)]
mod tests;

fn main() {
    let code = "(1 + (5 - 1)) == 5";

    for token in create_lexer(code){
        dbg!(token);
    }    
}
