use logos::Logos;
use nom::combinator::map;
use nom::error::ErrorKind;
use nom::error::ParseError;
use nom::IResult;
use nom_rule::rule;
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

type Input<'a> = &'a [Token<'a>];

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

fn ident<'a, Error: ParseError<Input<'a>>>(i: Input<'a>) -> IResult<Input<'a>, &str, Error> {
    map(satisfy(|token| token.kind == TokenKind::Ident), |token| {
        token.text
    })(i)
}

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

#[derive(Debug)]
#[allow(de)]
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

    let res: IResult<Input, CreateTableStmt> = create_table(&tokens);
    assert_eq!(
        format!("{res:?}"),
        r#"Ok(([], CreateTableStmt { name: "users", columns: [("id", "INT"), ("name", "VARCHAR")] }))"#
    );
}
