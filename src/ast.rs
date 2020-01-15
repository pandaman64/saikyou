#[derive(Clone, Copy, Debug)]
pub enum OpKind {
    Add,
    Minus,
    Mul,
    Div,
}

#[derive(Clone, Debug)]
pub enum Expr<'s> {
    IntLit(i32),
    Var(&'s str),
    BinOp(OpKind, Box<Self>, Box<Self>),
    Call(Box<Self>, Vec<Self>),
    Let(&'s str, Box<Self>, Box<Self>),
}

#[derive(Debug)]
pub struct Function<'s> {
    pub name: &'s str,
    pub arguments: Vec<&'s str>,
    pub body: Expr<'s>,
}
