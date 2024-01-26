#[macro_use]
extern crate syn;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn needs_feature(args: TokenStream, input: TokenStream) -> TokenStream {
    panic!("args: {:#?}, input: {:#?}", args, input);
}
