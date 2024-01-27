//! Compiler Builtins

pub static MARKERS: &[(&'static str, crate::ast::BuiltinMarkerFunc)] = &[
    ("transparent", _marker_transparent),
    ("feature", _marker_feature),
    ("__global_feature", _global_marker_feature),
    ("__global_no_compiler_builtins", |_, _, _| {}),
];

use std::ops::DerefMut;

use crate::{
    ast::{Context, Item, Value},
    error_and_return, pipeline_send, str_or_format,
};

use macros::flakc_builtin;

fn _marker_transparent(mut item: &mut Item, ctx: &mut Context, args: Vec<Value>) {
    if let Item::Module {
        ref mut is_transparent,
        ..
    } = item
    {
        *is_transparent = true;
    } else {
        pipeline_send!(
            #[Error]
            "Only inline modules can be marked as transparent.",
            "error originates from \"<builtin-markers>.transparent\"!"
        )
    }
}

fn _marker_feature(mut item: &mut Item, ctx: &mut Context, args: Vec<Value>) {
    let feature = match *args[0] {
        crate::token::Token::String(ref s) => s.clone(),
        ref t => error_and_return!(
            #[Error]
            "mismatched types.",
            ("Expected string literal found {:?}", t)
        ),
    };

    if !ctx.check_feature_gate(feature.as_str()) {
        *item = Item::_Disabled;
    }
}

fn _global_marker_feature(_: &mut Item, ctx: &mut Context, args: Vec<Value>) {
    for value in args.iter() {
        let feature = match **value {
            crate::token::Token::String(ref s) => s.clone(),
            ref t => error_and_return!(
                #[Error]
                "mismatched types.",
                ("Expected string literal found {:?}", t)
            ),
        };

        ctx.toggle_feature_gate(feature.as_str());
    }
}
