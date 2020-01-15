//! alpha renaming

use std::collections::HashMap;

use crate::ast::*;
use crate::parser::ParserExt;

type ParserExpr<'s> = Expr<ParserExt<'s>>;
type ParserFunc<'s> = Function<ParserExt<'s>>;

pub type AlphaExpr = Expr<AlphaExt>;
pub type AlphaFunc = Function<AlphaExt>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DefId(pub usize);

#[derive(Clone, Debug)]
pub enum AlphaExt {}

impl AstExt for AlphaExt {
    type Var = DefId;
    type BinOp = ();
    type Call = ();
    type Let = DefId;

    type FunName = DefId;
    type FunArg = DefId;
    type FunExt = ();
}

#[derive(Default, Debug)]
pub struct Environment<'s> {
    map: HashMap<&'s str, DefId>,
}

impl<'s> Environment<'s> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter<F, R>(&mut self, k: &'s str, v: DefId, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old = self.map.insert(k, v);

        // TODO: panic handling
        let result = f(self);

        if let Some(old) = old {
            self.map.insert(k, old);
        } else {
            self.map.remove(k);
        }

        result
    }

    pub fn insert(&mut self, k: &'s str, v: DefId) -> Option<DefId> {
        self.map.insert(k, v)
    }

    pub fn remove(&mut self, k: &'s str) -> Option<DefId> {
        self.map.remove(k)
    }
}

#[derive(Default)]
pub struct AlphaContext {
    current: usize,
}

impl AlphaContext {
    pub fn new() -> Self {
        Self::default()
    }

    fn new_def_id(&mut self) -> DefId {
        let ret = self.current;
        self.current += 1;
        DefId(ret)
    }

    pub fn rename_expr<'s>(
        &mut self,
        env: &mut Environment<'s>,
        expr: ParserExpr<'s>,
    ) -> anyhow::Result<AlphaExpr> {
        use Expr::*;

        Ok(match expr {
            IntLit(v) => IntLit(v),
            Var(x) => Var(*env
                .map
                .get(x)
                .ok_or_else(|| anyhow::anyhow!("{} not found in {:?}", x, env))?),
            BinOp(kind, lhs, rhs, ()) => BinOp(
                kind,
                Box::new(self.rename_expr(env, *lhs)?),
                Box::new(self.rename_expr(env, *rhs)?),
                (),
            ),
            Call(f, xs, ()) => Call(
                Box::new(self.rename_expr(env, *f)?),
                xs.into_iter()
                    .map(|x| self.rename_expr(env, x))
                    .collect::<anyhow::Result<Vec<_>>>()?,
                (),
            ),
            Let(x, e, body) => {
                let e = self.rename_expr(env, *e)?;

                let new_id = self.new_def_id();
                let body = env.enter(x, new_id, move |env| self.rename_expr(env, *body))?;
                Let(new_id, Box::new(e), Box::new(body))
            }
        })
    }

    pub fn rename_func<'s>(
        &mut self,
        env: &mut Environment<'s>,
        f: ParserFunc<'s>,
    ) -> anyhow::Result<AlphaFunc> {
        // enter for each argument, then rename body
        fn go<'s>(
            i: usize,
            len: usize,
            this: &mut AlphaContext,
            env: &mut Environment<'s>,
            args: &[&'s str],
            arg_id: &[DefId],
            body: ParserExpr<'s>,
        ) -> anyhow::Result<AlphaExpr> {
            if i == len {
                this.rename_expr(env, body)
            } else {
                env.enter(args[i], arg_id[i], move |env| {
                    go(i + 1, len, this, env, args, arg_id, body)
                })
            }
        }

        let name_id = self.new_def_id();
        let arg_id = f
            .arguments
            .iter()
            .map(|_| self.new_def_id())
            .collect::<Vec<_>>();
        let arg_id_ref = &arg_id;

        let body = env.enter(f.name, name_id, move |env| {
            go(
                0,
                arg_id_ref.len(),
                self,
                env,
                &f.arguments,
                arg_id_ref,
                f.body,
            )
        })?;

        Ok(Function {
            name: name_id,
            arguments: arg_id,
            body,
            ext: (),
        })
    }
}
