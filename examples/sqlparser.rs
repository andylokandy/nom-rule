use logos::Logos;
use nom::combinator::map;
use nom::error::ErrorKind;
use nom::error::ParseError;
use nom::Parser;
use nom::{IResult, Needed};
use nom_rule::rule;
use std::iter::{Cloned, Enumerate};
use std::slice::Iter;
use TokenKind::*;

#[derive(Logos, Clone, Copy, Debug, PartialEq)]
enum TokenKind {
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Whitespace,
    #[token("CREATE", ignore(ascii_case))]
    CREATE,
    #[token("TABLE", ignore(ascii_case))]
    TABLE,
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
    span: std::ops::Range<usize>,
}

#[derive(Debug, Clone)]
struct Input<'a>(&'a [Token<'a>]);

impl<'a> nom::Input for Input<'a> {
    type Item = Token<'a>;
    type Iter = Cloned<Iter<'a, Token<'a>>>;
    type IterIndices = Enumerate<Self::Iter>;

    fn input_len(&self) -> usize {
        self.0.len()
    }

    fn take(&self, index: usize) -> Self {
        Input(&self.0[0..index])
    }

    fn take_from(&self, index: usize) -> Self {
        Input(&self.0[index..])
    }

    fn take_split(&self, index: usize) -> (Self, Self) {
        let (prefix, suffix) = self.0.split_at(index);
        (Input(suffix), Input(prefix))
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.0.iter().position(|b| predicate(b.clone()))
    }

    fn iter_elements(&self) -> Self::Iter {
        self.0.iter().cloned()
    }

    fn iter_indices(&self) -> Self::IterIndices {
        self.iter_elements().enumerate()
    }

    fn slice_index(&self, count: usize) -> Result<usize, Needed> {
        if self.0.len() >= count {
            Ok(count)
        } else {
            Err(Needed::new(count - self.0.len()))
        }
    }
}

fn tokenise(input: &str) -> Vec<Token> {
    let mut lex = TokenKind::lexer(input);
    let mut tokens = Vec::new();

    while let Some(Ok(kind)) = lex.next() {
        tokens.push(Token {
            kind,
            text: lex.slice(),
            span: lex.span(),
        })
    }

    tokens
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

fn ident<'a, Error: ParseError<Input<'a>>>(i: Input<'a>) -> IResult<Input<'a>, &'a str, Error> {
    map(satisfy(|token| token.kind == Ident), |token| token.text).parse(i)
}

fn satisfy<'a, F, Error: ParseError<Input<'a>>>(
    cond: F,
) -> impl Fn(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>, Error>
where
    F: Fn(&Token<'a>) -> bool,
{
    move |i| match i.0.get(0).map(|t| {
        let b = cond(&t);
        (t, b)
    }) {
        Some((t, true)) => Ok((Input(&i.0[1..]), t)),
        _ => Err(nom::Err::Error(Error::from_error_kind(
            i,
            ErrorKind::Satisfy,
        ))),
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct CreateTableStmt {
    name: String,
    columns: Vec<(String, String)>,
}

fn main() {
    let tokens = tokenise("CREATE TABLE users (id INT, name VARCHAR);");

    let mut create_table = map(
        rule! {
            CREATE ~ TABLE ~ #ident ~ ^"(" ~ (#ident ~ #ident ~ ","?)* ~ ")" ~ ";"
            : "CREATE TABLE statement"
        },
        |(_create, _table, name, _lparan, columns, _rparan, _semicolon)| CreateTableStmt {
            name: name.to_string(),
            columns: columns
                .iter()
                .map(|(name, ty, _comma)| (name.to_string(), ty.to_string()))
                .collect(),
        },
    );

    let result: IResult<Input, CreateTableStmt> = create_table.parse(Input(&tokens));
    let (rest, stmt) = result.unwrap();

    println!("{stmt:?}");

    assert!(rest.0.is_empty());
    assert_eq!(stmt.name, "users");
    assert_eq!(
        &stmt.columns,
        &[
            ("id".to_string(), "INT".to_string()),
            ("name".to_string(), "VARCHAR".to_string())
        ]
    );
}
