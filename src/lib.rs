//! Flake Language
#![allow(dead_code)]
#![allow(warnings)]
#![feature(
    coroutines,
    iter_from_coroutine,
    coroutine_trait,
    generic_arg_infer,
    exclusive_range_pattern,
    slice_pattern,
    proc_macro_hygiene,
    extend_one,
    stmt_expr_attributes,
    exit_status_error,
    decl_macro
)]

/// The Flake Compiler version.
const FLAKEC_VERSION: &'static str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "ast")]
/// Abstract Syntax Tree.
pub mod ast;
#[cfg(feature = "llvm-codegen")]
/// LLVM Code-generation.
pub mod codegen;
#[cfg(feature = "eval")]
/// Compile-time Evaluation for Flake.
pub mod eval;
#[cfg(feature = "lexer")]
/// Flake's Lexer.
pub mod lexer;

/// Token Types.
pub mod token;

#[cfg(feature = "ast")]
/// Error Pipeline
pub mod pipeline;

#[cfg(feature = "ast")]
/// Error Pipeline
pub mod feature;

// Shared
#[path = "shared.rs"]
mod shared;

pub use shared::*;

#[cfg(feature = "ast")]
mod builtins;
