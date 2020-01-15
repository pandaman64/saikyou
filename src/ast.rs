use std::fmt::Debug;

// TODO: Clone + Debug bound of the AstExt itself is not required actually,
// but I want derived impls^^
pub trait AstExt<'s>: Clone + Debug {
    type Var: Clone + Debug;
    type BinOp: Clone + Debug;
    type Call: Clone + Debug;
    type Let: Clone + Debug;
}

#[derive(Clone, Copy, Debug)]
pub enum OpKind {
    Add,
    Minus,
    Mul,
    Div,
}

#[derive(Clone, Debug)]
pub enum Expr<'s, Ext: AstExt<'s>> {
    IntLit(i32),
    Var(&'s str, Ext::Var),
    BinOp(OpKind, Box<Self>, Box<Self>, Ext::BinOp),
    Call(Box<Self>, Vec<Self>, Ext::Call),
    Let(&'s str, Box<Self>, Box<Self>, Ext::Let),
}

#[derive(Debug)]
pub struct Function<'s, Ext: AstExt<'s>> {
    pub name: &'s str,
    pub arguments: Vec<&'s str>,
    pub body: Expr<'s, Ext>,
}
