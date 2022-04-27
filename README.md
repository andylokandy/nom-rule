# nom-rule

[![Documentation](https://docs.rs/nom-rule/badge.svg)](https://docs.rs/nom-rule/)
[![Crates.io](https://img.shields.io/crates/v/nom-rule.svg)](https://crates.io/crates/nom-rule)
[![LICENSE](https://img.shields.io/github/license/andylokandy/nom-rule.svg)](https://github.com/andylokandy/nom-rule/blob/master/LICENSE)

A procedural macro for defining [nom](https://crates.io/crates/nom) combinators in simple DSL. Requires `nom` v5.0+.

## Dependencies

```toml
[dependencies]
nom = "7"
nom-rule = "0.2"
```

## Syntax

The procedural macro `rule!` provided by this crate is designed for the ease of writing grammar spec as well as to improve maintainability, it follows these simple rules:

1. `TOKEN`: match the token by token kind. You should provide a parser to eat the next token if the token kind matched. it will get expanded into `match_token(TOKEN)`.
2. `";"`: match the token by token text. You should provide a parser to eat the next token if the token text matched. it will get expanded into `match_text(";")` in this example.
3. `#fn_name`: an external nom parser function. In the example above, `ident` is a predefined parser for identifiers.
4. `a ~ b ~ c`: a sequence of parsers to take one by one. It'll get expanded into `nom::sequence::tuple`.
5. `(...)+`: one or more repeated patterns. It'll get expanded into `nom::multi::many1`.
6. `(...)*`: zero or more repeated patterns. It'll get expanded into `nom::multi::many0`.
7. `(...)?`: Optional parser. It'll get expanded into `nom::combinator::opt`.
8. `a | b | c`: Choices between a, b, and c. It'll get expanded into `nom::branch::alt`.
9. `&a`: Positive predicate. It'll get expanded into `nom::combinator::map(nom::combinator::peek(a), |_| ())`. Note that it doesn't consume the input.
10. `!a`: Negative predicate. It'll get expanded into `nom::combinator::not`. Note that it doesn't consume the input.
11. `^a`:  Cut parser. It'll get expanded into `nom::combinator::cut`.
12. `... : "description"`: Context description for error reporting. It'll get expanded into `nom::error::context`.

## Example

Define `match_text` parser and `match_token` parser for your custom token type. You can use `nom::combinator::fail` as `match_token` if your parser use `&str` or `&[u8]` as input because you won't match on token kinds.

```rust
#[derive(Clone, Debug, PartialEq)]
struct Token<'a> {
    kind: TokenKind,
    text: &'a str,
    span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TokenKind {
    Whitespace,

    // Keywords
    CREATE,
    TABLE,

    // Symbols
    LParen,
    RParen,
    Semicolon,
    Comma,

    Ident,
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
```

Then give the two parser to `nom_rule::rule!` by wrapping it into a custom macro:

```rust
macro_rules! rule {
    ($($tt:tt)*) => { 
        nom_rule::rule!($crate::match_text, $crate::match_token, $($tt)*)
    }
}
```

To define a parser for the SQL of creating table:

```rust
let mut rule = rule!(
    CREATE ~ TABLE ~ #ident ~ ^"(" ~ (#ident ~ #ident ~ ","?)* ~ ")" ~ ";" : "CREATE TABLE statement"
);
```

It will get expanded into:

```rust
let mut rule = 
    nom::error::context(
        "CREATE TABLE statement",
        nom::sequence::tuple((
            (crate::match_token)(CREATE),
            (crate::match_token)(TABLE),
            ident,
            (nom::combinator::cut(crate::match_text)("(")),
            nom::multi::many0(nom::sequence::tuple((
                ident,
                ident,
                nom::combinator::opt((crate::match_text)(",")),
            ))),
            (crate::match_text)(")"),
            (crate::match_text)(";"),
        ))
    );
```

## Auto Sequence (nightly only)

`nom-rule` is able to automatically insert `~` in the rule when necessary so that you get the example above working the same as the following:

```rust
let mut rule = rule!(
    CREATE TABLE #ident "(" (#ident #ident ","?)* ")" ";" : "CREATE TABLE statement"
);
```

To enable this feature, you need to use a nightly channel rust complier, and add this to the `Cargo.toml`:

```toml
nom-rule = { version = "0.2", features = ["auto-sequence"] }
```
