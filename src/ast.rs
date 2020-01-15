use std::fmt::Debug;

// TODO: Clone + Debug bound of the AstExt itself is not required actually,
// but I want derived impls^^
pub trait AstExt: Clone + Debug {
    type Var: Clone + Debug;
    type BinOp: Clone + Debug;
    type Call: Clone + Debug;
    type Let: Clone + Debug;

    type FunName: Clone + Debug;
    type FunArg: Clone + Debug;
    type FunExt: Clone + Debug;
}

#[derive(Clone, Copy, Debug)]
pub enum OpKind {
    Add,
    Minus,
    Mul,
    Div,
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use OpKind::*;

        match self {
            Add => write!(f, "+"),
            Minus => write!(f, "-"),
            Mul => write!(f, "*"),
            Div => write!(f, "/"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr<Ext: AstExt> {
    IntLit(i32),
    Var(Ext::Var),
    BinOp(OpKind, Box<Self>, Box<Self>, Ext::BinOp),
    Call(Box<Self>, Vec<Self>, Ext::Call),
    Let(Ext::Let, Box<Self>, Box<Self>),
}

#[derive(Debug)]
pub struct Function<Ext: AstExt> {
    pub name: Ext::FunName,
    pub arguments: Vec<Ext::FunArg>,
    pub body: Expr<Ext>,
    pub ext: Ext::FunExt,
}
