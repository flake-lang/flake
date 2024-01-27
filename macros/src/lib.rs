#[macro_use]
extern crate syn;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn needs_feature(args: TokenStream, input: TokenStream) -> TokenStream {
    panic!("args: {:#?}, input: {:#?}", args, input);
}

#[proc_macro_attribute]
pub fn flakc_builtin(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}
