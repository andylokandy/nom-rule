# nom-rule

[![Documentation](https://docs.rs/nom-rule/badge.svg)](https://docs.rs/nom-rule/)
[![Crates.io](https://img.shields.io/crates/v/nom-rule.svg)](https://crates.io/crates/nom-rule)
[![LICENSE](https://img.shields.io/github/license/andylokandy/nom-rule.svg)](https://github.com/andylokandy/nom-rule/blob/master/LICENSE)

`nom-rule` is a small procedural macro for composing `nom` parsers with
grammar-shaped expressions.

The macro does not tokenize input and does not replace `nom`. It expands a rule
into ordinary `nom` parsers. You provide terminal matchers for your input type,
then use `rule!` to describe how those parsers fit together.

## Quick Start

Add both crates to your project. The generated code refers to `nom::...`, so
`nom` must be a direct dependency.

```toml
[dependencies]
nom = "8"
nom-rule = "0.5"
```

```rust
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, digit1};
use nom::combinator::map_res;
use nom::error::ParseError;
use nom::{IResult, Parser};
use nom_rule::rule;

fn match_text<'a, Error>(
    text: &'static str,
) -> impl Parser<&'a str, Output = &'a str, Error = Error>
where
    Error: ParseError<&'a str>,
{
    tag(text)
}

fn number(input: &str) -> IResult<&str, u32> {
    map_res(digit1, str::parse).parse(input)
}

fn assignment(input: &str) -> IResult<&str, (&str, u32)> {
    let mut parser = rule! {
        "let " ~ #alpha1 ~ " = " ~ #number ~ ";"
    };

    let (rest, (_, name, _, value, _)) = parser.parse(input)?;
    Ok((rest, (name, value)))
}

fn main() {
    let (rest, (name, value)) = assignment("let answer = 42;").unwrap();

    assert_eq!(rest, "");
    assert_eq!(name, "answer");
    assert_eq!(value, 42);
}
```

String literals call `match_text`. Parsers prefixed with `#` are ordinary `nom`
parsers or parser factories. A sequence returns a tuple, so the example maps out
the punctuation it does not need.

## Terminals

By default, `rule!` expects these names to be in scope when the corresponding
terminal form is used:

| Rule form | Expanded call |
|-----------|---------------|
| `"text"`  | `match_text("text")` |
| `TOKEN` or `path::TOKEN` | `match_token(TOKEN)` or `match_token(path::TOKEN)` |

The matcher signatures are up to your input type. They only need to return
something that implements `nom::Parser`.

You can also pass custom matcher paths before the rule:

```rust
let parser = rule! {
    tokens::text, tokens::kind,
    TokenKind::Create ~ #ident
};
```

## Rule Syntax

| Form | Meaning | Output shape |
|------|---------|--------------|
| `"text"` | Match text with `match_text` | matcher output |
| `TOKEN` | Match a token with `match_token` | matcher output |
| `#parser` | Use an existing parser | parser output |
| `#parser(args...)` | Call a parser factory | parser output |
| `a ~ b` | Run parsers in sequence | `(A, B)` |
| `a \| b` | Try alternatives with `nom::branch::alt` | branch output |
| `a?` | Optional parser | `Option<A>` |
| `a*` | Zero or more matches | `Vec<A>` |
| `a+` | One or more matches | `Vec<A>` |
| `&a` | Peek without consuming input | `A` |
| `!a` | Negative lookahead | `()` |
| `^a` | Cut on failure | `A` |
| `a : "context"` | Add `nom::error::context` | `A` |

Use parentheses to group subrules.

Operator precedence, from low to high:

1. choice: `a | b`
2. context: `a : "context"`
3. sequence: `a ~ b`
4. postfix operators: `?`, `*`, `+`
5. prefix operators: `&`, `!`, `^`

Choice branches must return compatible output shapes. An optional branch is
rejected before later alternatives because it can succeed without consuming
input.

## Complete Example

See [`examples/sqlparser.rs`](examples/sqlparser.rs) for a token-stream parser
that lexes a small SQL statement and parses it with `rule!`.
