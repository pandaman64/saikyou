//! type inference & checking

use std::collections::HashMap;

use crate::alpha::{AlphaExpr, AlphaFunc, DefId};
use crate::ast::*;

type TypedExpr = Expr<TypeExt>;
type TypedFunc = Function<TypeExt>;

#[derive(Clone, Debug)]
pub enum TypeExt {}

impl AstExt for TypeExt {
    type Var = (DefId, TypeId);
    type BinOp = TypeId;
    type Call = TypeId;
    type Let = (DefId, TypeId, TypeId); // binder, type of binder, type of body

    type FunName = DefId;
    type FunArg = DefId;
    type FunExt = TypeId;
}

impl TypedExpr {
    fn ty(&self) -> TypeId {
        use Expr::*;

        match self {
            IntLit(_) => INT,
            Var((_, tid)) | BinOp(_, _, _, tid) | Call(_, _, tid) | Let((_, _, tid), _, _) => *tid,
        }
    }

    pub fn pprint(&self, ctx: &TypeContext) -> String {
        use Expr::*;

        match self {
            IntLit(v) => v.to_string(),
            Var((name, tid)) => format!("(`{}`: {})", name.0, tid.pprint(ctx)),
            BinOp(kind, lhs, rhs, tid) => format!(
                "({} {} {}: {})",
                lhs.pprint(ctx),
                kind,
                rhs.pprint(ctx),
                tid.pprint(ctx)
            ),
            Call(f, args, tid) => format!(
                "({} {:?}: {})",
                f.pprint(ctx),
                args.iter().map(|arg| arg.pprint(ctx)).collect::<Vec<_>>(),
                tid.pprint(ctx)
            ),
            Let((name, vty, bodyty), e, body) => format!(
                "(let `{}`: {} = {};\n{}: {})",
                name.0,
                vty.pprint(ctx),
                e.pprint(ctx),
                body.pprint(ctx),
                bodyty.pprint(ctx)
            ),
        }
    }
}

struct DisplayTypedFunc<'f, 'c>(&'f TypedFunc, &'c TypeContext);

impl std::fmt::Display for DisplayTypedFunc<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let DisplayTypedFunc(func, ctx) = self;
        match &ctx.types[func.ext.0] {
            Type::Fun(argsty, bodyty) => {
                write!(f, "fun `{}` ", func.name.0)?;
                for (arg, argty) in func.arguments.iter().zip(argsty.iter()) {
                    write!(f, "(`{}`: {}) ", arg.0, argty.pprint(ctx))?;
                }
                write!(f, ": {} =\n\t", bodyty.pprint(ctx))?;

                write!(f, "{}", func.body.pprint(ctx))
            }
            _ => unreachable!("malformed function: {:?}", func),
        }
    }
}

impl TypedFunc {
    pub fn pprint(&self, ctx: &TypeContext) -> String {
        format!("{}", DisplayTypedFunc(self, ctx))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TypeId(pub usize);

impl TypeId {
    pub fn pprint(&self, ctx: &TypeContext) -> String {
        ctx.types[self.0].pprint(ctx)
    }
}

// id of simple types
const INT: TypeId = TypeId(0);

const PLACEHOLDER: TypeId = TypeId(std::usize::MAX);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simple_types() {
        let ctx = TypeContext::new();
        assert!(matches!(ctx.types[INT.0], Type::Int));
    }
}

#[derive(Clone, Debug)]
pub enum Type {
    Int,
    Var(TypeId),
    Fun(Vec<TypeId>, TypeId),
}

impl Type {
    pub fn pprint(&self, ctx: &TypeContext) -> String {
        use Type::*;

        match self {
            Type::Int => "int".to_string(),
            Var(idx) => {
                let root = ctx.get_root(*idx);
                if root.0 == idx.0 {
                    format!("\\{}", idx.0)
                } else {
                    ctx.types[root.0].pprint(ctx)
                }
            }
            Fun(args, body) => format!(
                "{:?} -> {}",
                args.iter().map(|arg| arg.pprint(ctx)).collect::<Vec<_>>(),
                body.pprint(ctx)
            ),
        }
    }
}

#[derive(Debug, Default)]
pub struct Environment {
    map: HashMap<DefId, TypeId>,
}

impl Environment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter<F, R>(&mut self, k: DefId, v: TypeId, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old = self.map.insert(k, v);

        // TODO: panic handling
        let result = f(self);

        if let Some(old) = old {
            self.map.insert(k, old);
        } else {
            self.map.remove(&k);
        }

        result
    }

    pub fn insert(&mut self, k: DefId, v: TypeId) -> Option<TypeId> {
        self.map.insert(k, v)
    }

    pub fn remove(&mut self, k: DefId) -> Option<TypeId> {
        self.map.remove(&k)
    }
}

#[derive(Debug)]
pub struct TypeContext {
    types: Vec<Type>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self {
            types: vec![Type::Int],
        }
    }

    pub fn new_type_var(&mut self) -> TypeId {
        let id = TypeId(self.types.len());
        self.types.push(Type::Var(id));
        id
    }

    pub fn new_type(&mut self, ty: Type) -> TypeId {
        let id = TypeId(self.types.len());
        self.types.push(ty);
        id
    }

    fn get_root(&self, ty: TypeId) -> TypeId {
        use Type::*;

        match self.types[ty.0] {
            Int => INT,
            Var(idx) => {
                if ty.0 == idx.0 {
                    ty
                } else {
                    self.get_root(idx)
                }
            }
            Fun(_, _) => ty,
        }
    }

    fn resolve(&mut self, ty: TypeId) -> TypeId {
        use Type::*;

        match std::mem::replace(&mut self.types[ty.0], Var(PLACEHOLDER)) {
            Int => {
                self.types[ty.0] = Int;
                INT
            }
            Var(idx) => {
                assert_ne!(idx.0, PLACEHOLDER.0, "cycle detected: {:?}", self);

                // pointing to itself, this is root
                if idx.0 == ty.0 {
                    self.types[ty.0] = Var(idx);
                    return ty;
                }

                // compression
                let root = self.resolve(idx);
                self.types[ty.0] = Var(root);
                root
            }
            Fun(mut args, body) => {
                // compression
                for arg in args.iter_mut() {
                    *arg = self.resolve(*arg);
                }
                let body = self.resolve(body);

                self.types[ty.0] = Fun(args, body);
                ty
            }
        }
    }

    fn unify(&mut self, ilhs: TypeId, irhs: TypeId) -> anyhow::Result<()> {
        use std::mem::replace;
        use Type::*;

        let ilhs = self.resolve(ilhs);
        let irhs = self.resolve(irhs);

        if ilhs.0 == irhs.0 {
            return Ok(());
        }

        let mut lhs = replace(&mut self.types[ilhs.0], Var(PLACEHOLDER));
        let mut rhs = replace(&mut self.types[irhs.0], Var(PLACEHOLDER));

        match (&mut lhs, &mut rhs) {
            (Int, Int) => {}
            (Fun(largs, lbody), Fun(rargs, rbody)) => {
                if largs.len() != rargs.len() {
                    return Err(anyhow::anyhow!(
                        "arity mismatch: {:?} vs {:?}",
                        largs,
                        rargs
                    ));
                }

                for (larg, rarg) in largs.iter_mut().zip(rargs.iter_mut()) {
                    self.unify(*larg, *rarg)?;
                }

                self.unify(*lbody, *rbody)?;
            }
            (Var(lidx), Var(_ridx)) => {
                *lidx = irhs;
            }
            (Var(lidx), _) => *lidx = irhs,
            (_, Var(ridx)) => *ridx = ilhs,
            _ => return Err(anyhow::anyhow!("type mismatch: {:?} vs {:?}", lhs, rhs)),
        }

        self.types[ilhs.0] = lhs;
        self.types[irhs.0] = rhs;

        Ok(())
    }

    pub fn infer_expr(
        &mut self,
        env: &mut Environment,
        expr: AlphaExpr,
    ) -> anyhow::Result<TypedExpr> {
        use Expr::*;

        Ok(match expr {
            IntLit(v) => IntLit(v),
            Var(x) => {
                // lookup type of `x`
                let ty = env
                    .map
                    .get(&x)
                    .ok_or_else(|| anyhow::anyhow!("{:?} not found in {:?}", x, env))?;
                Var((x, self.resolve(*ty)))
            }
            BinOp(kind, lhs, rhs, ()) => {
                let lhs = self.infer_expr(env, *lhs)?;
                let rhs = self.infer_expr(env, *rhs)?;

                // currently, support only integer operators
                self.unify(lhs.ty(), INT)?;
                self.unify(rhs.ty(), INT)?;

                BinOp(kind, Box::new(lhs), Box::new(rhs), INT)
            }
            Call(f, args, ()) => {
                let f = self.infer_expr(env, *f)?;
                let args = args
                    .into_iter()
                    .map(|arg| self.infer_expr(env, arg))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let retty = self.new_type_var();
                let fty = self.new_type(Type::Fun(args.iter().map(TypedExpr::ty).collect(), retty));

                self.unify(f.ty(), fty)?;

                Call(Box::new(f), args, retty)
            }
            Let(x, e, body) => {
                let e = self.infer_expr(env, *e)?;

                let xty = self.new_type_var();
                self.unify(xty, e.ty())?;

                let body = env.enter(x, xty, move |env| self.infer_expr(env, *body))?;

                Let((x, xty, body.ty()), Box::new(e), Box::new(body))
            }
        })
    }

    pub fn infer_func(
        &mut self,
        env: &mut Environment,
        func: AlphaFunc,
    ) -> anyhow::Result<TypedFunc> {
        fn go(
            i: usize,
            len: usize,
            this: &mut TypeContext,
            env: &mut Environment,
            arg_ty: Vec<TypeId>,
            func: AlphaFunc,
        ) -> anyhow::Result<TypedFunc> {
            if i == len {
                let body = this.infer_expr(env, func.body)?;

                let funty = this.new_type(Type::Fun(arg_ty, body.ty()));

                Ok(Function {
                    name: func.name,
                    arguments: func.arguments,
                    body,
                    ext: funty,
                })
            } else {
                env.enter(func.arguments[i], arg_ty[i], move |env| {
                    go(i + 1, len, this, env, arg_ty, func)
                })
            }
        }

        let len = func.arguments.len();
        let arg_ty = func.arguments.iter().map(|_| self.new_type_var()).collect();
        go(0, len, self, env, arg_ty, func)
    }
}
