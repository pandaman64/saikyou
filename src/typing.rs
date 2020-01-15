//! type inference & checking

use std::collections::{HashMap, HashSet};
use std::fmt;

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

struct DisplayTypedExpr<'e, 'c>(&'e TypedExpr, &'c TypeContext);

impl fmt::Display for DisplayTypedExpr<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Expr::*;

        let DisplayTypedExpr(expr, ctx) = self;

        match expr {
            IntLit(v) => write!(f, "{}", v),
            Var((name, tid)) => write!(f, "(`{}`: {})", name.0, DisplayType(*tid, ctx)),
            BinOp(kind, lhs, rhs, tid) => write!(
                f,
                "({} {} {}: {})",
                Self(lhs, ctx),
                kind,
                Self(rhs, ctx),
                DisplayType(*tid, ctx),
            ),
            Call(fun, args, tid) => {
                write!(f, "({}", Self(fun, ctx))?;
                for arg in args.iter() {
                    write!(f, " {}", Self(arg, ctx))?;
                }
                write!(f, ": {})", DisplayType(*tid, ctx))
            }
            Let((name, vty, bodyty), e, body) => write!(
                f,
                "(let `{}`: {} = {}\n{}: {})",
                name.0,
                DisplayType(*vty, ctx),
                Self(e, ctx),
                Self(body, ctx),
                DisplayType(*bodyty, ctx),
            ),
        }
    }
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
        format!("{}", DisplayTypedExpr(self, ctx))
    }
}

struct DisplayTypedFunc<'f, 'c>(&'f TypedFunc, &'c TypeContext);

impl fmt::Display for DisplayTypedFunc<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let DisplayTypedFunc(func, ctx) = self;
        match &ctx.types[func.ext.0] {
            Type::Fun(genargsty, argsty, bodyty) => {
                write!(f, "fun `{}`", func.name.0)?;
                if !genargsty.is_empty() {
                    write!(f, "[forall")?;
                    for genargty in genargsty.iter() {
                        write!(f, " {}", DisplayType(*genargty, ctx))?
                    }
                    write!(f, "]")?
                }
                for (arg, argty) in func.arguments.iter().zip(argsty.iter()) {
                    write!(f, "(`{}`: {}) ", arg.0, DisplayType(*argty, ctx))?;
                }
                write!(f, ": {} =\n\t", DisplayType(*bodyty, ctx))?;

                write!(f, "{}", DisplayTypedExpr(&func.body, ctx))
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

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
    ForallVar(TypeId),
    // forall T, (X, ..., Z) -> R
    Fun(HashSet<TypeId>, Vec<TypeId>, TypeId),
}

struct DisplayType<'c>(TypeId, &'c TypeContext);

impl fmt::Display for DisplayType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Type::*;

        let DisplayType(tid, ctx) = self;

        match &ctx.types[tid.0] {
            Int => write!(f, "int"),
            Var(idx) => {
                let root = ctx.get_root(*idx);
                if matches!(ctx.types[root.0], Var(v) if v == root) {
                    write!(f, "${}", root.0)
                } else {
                    write!(f, "{}", Self(root, ctx))
                }
            }
            ForallVar(idx) => write!(f, "#{}", idx.0),
            Fun(genargs, args, body) => {
                write!(f, "(")?;

                if !genargs.is_empty() {
                    write!(f, "[forall")?;
                    for genarg in genargs.iter() {
                        write!(f, " {}", Self(*genarg, ctx))?;
                    }
                    write!(f, "]")?;
                }

                for arg in args.iter() {
                    write!(f, "{} -> ", Self(*arg, ctx))?;
                }

                write!(f, "{})", Self(*body, ctx))
            }
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

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
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
            ForallVar(idx) => {
                assert_eq!(ty, idx);
                ty
            }
            Fun(_, _, _) => ty,
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
            ForallVar(idx) => {
                assert_eq!(ty, idx);
                self.types[ty.0] = ForallVar(idx);
                ty
            }
            Fun(genargs, mut args, body) => {
                // compression
                for arg in args.iter_mut() {
                    *arg = self.resolve(*arg);
                }
                let body = self.resolve(body);

                self.types[ty.0] = Fun(genargs, args, body);
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
            (ForallVar(_), _) | (_, ForallVar(_)) => {
                unreachable!("not instantiated: {:?} or {:?}\nctx: {:?}", lhs, rhs, self)
            }
            (Int, Int) => {}
            (Fun(lgenargs, largs, lbody), Fun(rgenargs, rargs, rbody)) => {
                assert!(lgenargs.is_empty() && rgenargs.is_empty());

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

    #[must_use]
    fn generalize_ty(&mut self, ty: TypeId) -> HashSet<TypeId> {
        use Type::*;

        let mut ret = HashSet::new();

        let root = self.resolve(ty);
        match std::mem::replace(&mut self.types[root.0], Var(PLACEHOLDER)) {
            Var(v) if v == ty => {
                ret.insert(root);
                self.types[root.0] = ForallVar(ty)
            }
            Fun(genargs, args, body) => {
                assert!(genargs.is_empty());

                for arg in args.iter() {
                    ret.extend(self.generalize_ty(*arg));
                }
                ret.extend(self.generalize_ty(body));

                self.types[root.0] = Fun(genargs, args, body);
            }
            t => {
                self.types[root.0] = t;
            }
        }

        ret
    }

    #[must_use]
    fn generalize(&mut self, expr: &TypedExpr) -> HashSet<TypeId> {
        use Expr::*;

        let mut ret = HashSet::new();

        match expr {
            IntLit(_) => {}
            Var(_) => ret.extend(self.generalize_ty(expr.ty())),
            BinOp(_, lhs, rhs, retty) => {
                ret.extend(self.generalize(lhs));
                ret.extend(self.generalize(rhs));
                ret.extend(self.generalize_ty(*retty));
            }
            Call(f, args, retty) => {
                ret.extend(self.generalize(f));
                for arg in args.iter() {
                    ret.extend(self.generalize(arg));
                }
                ret.extend(self.generalize_ty(*retty));
            }
            Let((_, binderty, bodyty), e, body) => {
                ret.extend(self.generalize(e));
                ret.extend(self.generalize(body));

                ret.extend(self.generalize_ty(*binderty));
                ret.extend(self.generalize_ty(*bodyty));
            }
        }

        ret
    }

    fn instantiate(&mut self, env: &mut HashMap<TypeId, TypeId>, ty: TypeId) -> TypeId {
        use Type::*;

        match std::mem::replace(&mut self.types[ty.0], Var(PLACEHOLDER)) {
            Int => {
                self.types[ty.0] = Int;
                ty
            }
            Var(tid) => {
                self.types[ty.0] = Var(tid);
                let root = self.resolve(tid);
                if root == ty {
                    ty
                } else {
                    self.instantiate(env, root)
                }
            }
            ForallVar(tid) => {
                assert_eq!(ty, tid);

                self.types[ty.0] = ForallVar(tid);
                // assign a fresh variable to this generalized type argument
                *env.entry(ty).or_insert_with(|| self.new_type_var())
            }
            // generalize forall arguments of fun
            Fun(genargs, args, body) => {
                for genarg in genargs.iter() {
                    env.entry(*genarg).or_insert_with(|| self.new_type_var());
                }

                let new_args = args.iter().map(|&arg| self.instantiate(env, arg)).collect();
                let new_body = self.instantiate(env, body);
                let new_funty = Fun(HashSet::new(), new_args, new_body);

                self.types[ty.0] = Fun(genargs, args, body);
                let ty = self.new_type(new_funty);
                assert!(
                    !self.contains_generalized(ty),
                    "instantiated = {}",
                    DisplayType(ty, self)
                );
                ty
            }
        }
    }

    fn contains_generalized(&self, ty: TypeId) -> bool {
        use Type::*;

        match &self.types[self.get_root(ty).0] {
            Int | Var(_) => false,
            ForallVar(_) => true,
            Fun(genargs, args, body) => {
                (!genargs.is_empty())
                    || args.iter().any(|ty| self.contains_generalized(*ty))
                    || self.contains_generalized(*body)
            }
        }
    }

    fn check_no_placeholder(&self) {
        for ty in self.types.iter() {
            if matches!(ty, Type::Var(p) if *p == PLACEHOLDER) {
                panic!("invalid type");
            }
        }
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
                    .map(|ty| self.instantiate(&mut HashMap::new(), *ty))
                    .ok_or_else(|| anyhow::anyhow!("{:?} not found in {:?}", x, env))?;
                assert!(
                    !self.contains_generalized(ty),
                    "{:?} contains generalized var while referring to {:?}\nctx: {:?}\nenv: {:?}",
                    ty,
                    x,
                    self,
                    env
                );

                self.check_no_placeholder();
                Var((x, self.resolve(ty)))
            }
            BinOp(kind, lhs, rhs, ()) => {
                let lhs = self.infer_expr(env, *lhs)?;
                let rhs = self.infer_expr(env, *rhs)?;

                // currently, support only integer operators
                self.unify(lhs.ty(), INT)?;
                self.unify(rhs.ty(), INT)?;

                self.check_no_placeholder();

                BinOp(kind, Box::new(lhs), Box::new(rhs), INT)
            }
            Call(f, args, ()) => {
                let f = self.infer_expr(env, *f)?;
                let args = args
                    .into_iter()
                    .map(|arg| self.infer_expr(env, arg))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let retty = self.new_type_var();
                let fty = self.new_type(Type::Fun(
                    HashSet::new(),
                    args.iter().map(TypedExpr::ty).collect(),
                    retty,
                ));
                for arg in args.iter() {
                    assert!(
                        !self.contains_generalized(arg.ty()),
                        "arg contains generalized var: args = {:?}",
                        args
                    );
                }
                assert!(
                    !self.contains_generalized(fty),
                    "fty contains generalized var: args = {:?}",
                    args
                );

                self.unify(f.ty(), fty)?;

                self.check_no_placeholder();

                Call(Box::new(f), args, retty)
            }
            Let(x, e, body) => {
                let e = self.infer_expr(env, *e)?;

                let xty = self.new_type_var();
                self.unify(xty, e.ty())?;

                self.check_no_placeholder();

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

                let funty = this.new_type(Type::Fun(HashSet::new(), arg_ty, body.ty()));

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
        let func = go(0, len, self, env, arg_ty, func)?;

        let body_genargs = self.generalize(&func.body);
        match std::mem::replace(&mut self.types[func.ext.0], Type::Var(PLACEHOLDER)) {
            Type::Fun(mut genargs, args, body) => {
                assert!(genargs.is_empty());

                genargs.extend(body_genargs);

                for arg in args.iter() {
                    genargs.extend(self.generalize_ty(*arg));
                }

                genargs.extend(self.generalize_ty(body));

                self.types[func.ext.0] = Type::Fun(genargs, args, body);
            }
            _ => unreachable!("must be Fun"),
        }

        self.check_no_placeholder();

        Ok(func)
    }
}
