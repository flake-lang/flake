//! Flake Language

#[allow(dead_code)]

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