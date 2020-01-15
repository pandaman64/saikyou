use nom::branch::*;
use nom::bytes::complete::*;
use nom::character::complete::*;
use nom::combinator::*;
use nom::error::VerboseError;
use nom::multi::*;
use nom::sequence::*;

use std::marker::PhantomData;

use crate::ast;
use crate::ast::{AstExt, OpKind};

// fun f x y =
//   let z = g x;
//   y + z.

#[derive(Clone, Debug)]
pub struct ParserExt<'s>(PhantomData<&'s ()>);

impl<'s> AstExt for ParserExt<'s> {
    type Var = &'s str;
    type BinOp = ();
    type Call = ();
    type Let = &'s str;

    type FunName = &'s str;
    type FunArg = &'s str;
}

type IResult<I, O> = nom::IResult<I, O, VerboseError<I>>;

type Expr<'s> = ast::Expr<ParserExt<'s>>;
type Function<'s> = ast::Function<ParserExt<'s>>;

fn identifier(input: &str) -> IResult<&str, &str> {
    // first must be the subset of the following
    let first = "abcdefghijklmnopqrstuvwxyz_";
    let following = "abcdefghijklmnopqrstuvwxyz0123456789_!?";

    // check if the first character satisfies the condition (not consume the character yet)
    let (input, _) = peek(one_of(first))(input)?;

    // take all of the identifier using more permissive set of characters
    is_a(following)(input)
}

fn argument(input: &str) -> IResult<&str, &str> {
    identifier(input)
}

fn expression(input: &str) -> IResult<&str, Expr> {
    fn primary(input: &str) -> IResult<&str, Expr> {
        fn paren(input: &str) -> IResult<&str, Expr> {
            let open = terminated(char('('), multispace0);
            let close = preceded(multispace0, char(')'));
            delimited(open, expression, close)(input)
        }

        fn var(input: &str) -> IResult<&str, Expr> {
            let (input, ident) = identifier(input)?;
            Ok((input, Expr::Var(ident)))
        }

        fn intlit(input: &str) -> IResult<&str, Expr> {
            let first = "123456789";
            let following = "0123456789";

            let (input, _) = peek(one_of(first))(input)?;
            let (input, s) = is_a(following)(input)?;

            Ok((input, Expr::IntLit(s.parse().unwrap())))
        }

        alt((paren, intlit, var))(input)
    }

    fn call(input: &str) -> IResult<&str, Expr> {
        let (input, f) = primary(input)?;

        let (input, args) = fold_many0(
            preceded(multispace1, primary),
            Vec::new(),
            |mut args, arg| {
                args.push(arg);
                args
            },
        )(input)?;

        if args.is_empty() {
            Ok((input, f))
        } else {
            Ok((input, Expr::Call(Box::new(f), args, ())))
        }
    }

    fn muldiv(input: &str) -> IResult<&str, Expr> {
        fn following(input: &str) -> IResult<&str, (OpKind, Expr)> {
            let (input, _) = multispace0(input)?;

            let (input, c) = alt((char('*'), char('/')))(input)?;
            let (input, _) = multispace0(input)?;

            let (input, rhs) = call(input)?;

            match c {
                '*' => Ok((input, (OpKind::Mul, rhs))),
                '/' => Ok((input, (OpKind::Div, rhs))),
                _ => unreachable!(),
            }
        }
        let (input, lhs) = call(input)?;

        // TODO: not ideal, because fold_many0 requires Clone
        fold_many0(following, lhs, |lhs, (op, rhs)| {
            Expr::BinOp(op, Box::new(lhs), Box::new(rhs), ())
        })(input)
    }

    fn addmin(input: &str) -> IResult<&str, Expr> {
        fn following(input: &str) -> IResult<&str, (OpKind, Expr)> {
            let (input, _) = multispace0(input)?;

            let (input, c) = alt((char('+'), char('-')))(input)?;
            let (input, _) = multispace0(input)?;

            let (input, rhs) = muldiv(input)?;

            match c {
                '+' => Ok((input, (OpKind::Add, rhs))),
                '-' => Ok((input, (OpKind::Minus, rhs))),
                _ => unreachable!(),
            }
        }
        let (input, lhs) = muldiv(input)?;

        // TODO: not ideal, because fold_many0 requires Clone
        fold_many0(following, lhs, |lhs, (op, rhs)| {
            Expr::BinOp(op, Box::new(lhs), Box::new(rhs), ())
        })(input)
    }

    fn let_(input: &str) -> IResult<&str, Expr> {
        let (input, _) = tag("let")(input)?;
        let (input, _) = multispace1(input)?;

        let (input, var) = identifier(input)?;
        let (input, _) = multispace0(input)?;

        let (input, _) = char('=')(input)?;
        let (input, _) = multispace0(input)?;

        let (input, e) = expression(input)?;
        let (input, _) = multispace0(input)?;

        let (input, _) = char(';')(input)?;
        let (input, _) = multispace0(input)?;

        let (input, body) = expression(input)?;

        Ok((input, Expr::Let(var, Box::new(e), Box::new(body))))
    }

    alt((let_, addmin))(input)
}

fn function(input: &str) -> IResult<&str, Function> {
    let (input, _) = tag("fun")(input)?;
    let (input, _) = multispace1(input)?;

    let (input, name) = identifier(input)?;
    let (input, _) = multispace1(input)?;

    let (input, arguments) = separated_list(multispace1, argument)(input)?;
    let (input, _) = multispace0(input)?;

    let (input, _) = char('=')(input)?;
    let (input, _) = multispace1(input)?;

    let (input, body) = expression(input)?;
    let (input, _) = multispace0(input)?;

    let (input, _) = char('.')(input)?;

    Ok((
        input,
        Function {
            name,
            arguments,
            body,
        },
    ))
}

pub fn program(input: &str) -> IResult<&str, Vec<Function>> {
    all_consuming(delimited(
        multispace0,
        separated_nonempty_list(multispace0, function),
        multispace0,
    ))(input)
}
