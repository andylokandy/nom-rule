use logos::{Logos, Span};
use nom::{
    combinator::map,
    error::{ErrorKind, ParseError},
    IResult,
};

use TokenKind::*;

macro_rules! rule {
    ($($tt:tt)*) => { nom_rule::rule!(match_text, match_token, $($tt)*) }
}

#[test]
fn sql_create_table() {
    let tokens = tokenise("create table user (id int, name varchar);");

    let mut rule = rule!(
        CREATE ~ TABLE ~ #ident ~ ^"(" ~ (#ident ~ #ident ~ ","?)* ~ ")" ~ ";" : "CREATE TABLE statement"
    );

    let res: IResult<_, _> = rule(&tokens);
    assert_eq!(
        res.unwrap().1,
        (
            &Token {
                kind: CREATE,
                text: "create",
                span: 0..6,
            },
            &Token {
                kind: TABLE,
                text: "table",
                span: 7..12,
            },
            "user",
            &Token {
                kind: LParen,
                text: "(",
                span: 18..19,
            },
            vec![
                (
                    "id",
                    "int",
                    Some(&Token {
                        kind: Comma,
                        text: ",",
                        span: 25..26,
                    },),
                ),
                ("name", "varchar", None,),
            ],
            &Token {
                kind: RParen,
                text: ")",
                span: 39..40,
            },
            &Token {
                kind: Semicolon,
                text: ";",
                span: 40..41,
            },
        ),
    );
}

#[cfg(feature = "auto-sequence")]
#[test]
fn sql_column_ref() {
    let tokens = tokenise("a.b");

    let mut rule = rule!(
         #ident ("." #ident)?
    );

    let res: IResult<_, _> = rule(&tokens);
    assert_eq!(
        dbg!(res.unwrap().1),
        (
            "a",
            Some((
                &Token {
                    kind: Whitespace,
                    text: ".",
                    span: 1..2,
                },
                "b",
            ),),
        ),
    );
}

#[derive(Logos, Clone, Copy, Debug, PartialEq)]
enum TokenKind {
    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Whitespace,

    // Keywords
    #[token("CREATE", ignore(ascii_case))]
    CREATE,
    #[token("TABLE", ignore(ascii_case))]
    TABLE,

    // Symbols
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,

    #[regex("[a-zA-Z][a-zA-Z0-9]*")]
    Ident,
}

#[derive(Clone, Debug, PartialEq)]
struct Token<'a> {
    kind: TokenKind,
    text: &'a str,
    span: Span,
}

fn tokenise(input: &str) -> Vec<Token> {
    let mut lex = TokenKind::lexer(input);
    let mut tokens = Vec::new();

    while let Some(kind) = lex.next() {
        tokens.push(Token {
            kind,
            text: lex.slice(),
            span: lex.span(),
        })
    }

    tokens
}

type Input<'a> = &'a [Token<'a>];

fn satisfy<'a, F, Error: ParseError<Input<'a>>>(
    cond: F,
) -> impl Fn(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>, Error>
where
    F: Fn(&Token<'a>) -> bool,
{
    move |i| match i.get(0).map(|t| {
        let b = cond(&t);
        (t, b)
    }) {
        Some((t, true)) => Ok((&i[1..], t)),
        _ => Err(nom::Err::Error(Error::from_error_kind(
            i,
            ErrorKind::Satisfy,
        ))),
    }
}

fn match_text<'a, Error: ParseError<Input<'a>>>(
    text: &'a str,
) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>, Error> {
    move |i| satisfy(|token: &Token<'a>| token.text == text)(i)
}

fn match_token<'a, Error: ParseError<Input<'a>>>(
    kind: TokenKind,
) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>, Error> {
    move |i| satisfy(|token: &Token<'a>| token.kind == kind)(i)
}

fn ident<'a, Error: ParseError<Input<'a>>>(i: Input<'a>) -> IResult<Input<'a>, &str, Error> {
    map(satisfy(|token| token.kind == TokenKind::Ident), |token| {
        token.text
    })(i)
}
