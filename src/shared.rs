#[cfg(all(feature = "ast", feature = "lexer"))]
mod __impl {
    use crate::ast::{self, *};
    use crate::lexer::*;
    use std::collections::HashMap;

    use crate::builtins as __builtins;

    pub fn compile_module(name: &str, code: &str) -> Module {
        let mut context = ast::Context {
            locals: HashMap::new(),
            can_return: true,
            functions: HashMap::new(),
            types: [].into(),
            feature_gates: {
                use crate::feature::FeatureKind::*;

                HashMap::from(include!("../features.specs"))
            },
            markers: HashMap::from_iter(
                __builtins::MARKERS
                    .iter()
                    .map(|(n, f)| ((*n).to_owned(), ast::MarkerImpl::BuiltIn(*f))),
            ),
        };

        let mut tokens = create_lexer(code);

        let mut tokens_peekable = tokens.peekable();

        let mut tree = Vec::<ast::Node>::new();

        loop {
            if tokens_peekable.peek() == None {
                break;
            }

            if let Some(ast_node) = ast::parse_node(&mut tokens_peekable, &mut context) {
                // eval::eval_statement(ast_node.clone(), &mut eval_context);
                //   println!("{:#?}", &ast_node);
                //  println!("TYPE = {:#?}", ast::infer_expr_type(ast_node.clone()));
                tree.push(ast_node);
            } else {
                break;
            }
        }

        Module::new(name, tree, context)
    }
}

#[cfg(all(feature = "ast", feature = "lexer"))]
pub use __impl::*;
