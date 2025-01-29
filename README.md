# nom-rule

[![Documentation](https://docs.rs/nom-rule/badge.svg)](https://docs.rs/nom-rule/)
[![Crates.io](https://img.shields.io/crates/v/nom-rule.svg)](https://crates.io/crates/nom-rule)
[![LICENSE](https://img.shields.io/github/license/andylokandy/nom-rule.svg)](https://github.com/andylokandy/nom-rule/blob/master/LICENSE)

A procedural macro for writing nom parsers using a grammar-like syntax.

## Overview

The `nom-rule` crate provides the `rule!` macro, which allows you to define nom parsers using a domain-specific language (DSL) that resembles grammar rules. This makes your parsers more readable and easier to maintain.

## Features

- Define parsers using a grammar-like syntax.
- Integrates seamlessly with `nom` combinators.
- Supports sequences, choices, repetitions, optional parsing, and more.

## Usage

```rust
use nom_rule::rule;

// Define your match functions
fn match_text<'a>(text: &'a str) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>> {
    // Implementation
}

fn match_token<'a>(kind: TokenKind) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>> {
    // Implementation
}

fn ident<'a>(input: Input<'a>) -> IResult<Input<'a>, &str> {
    // Implementation
}

fn main() {
    // Use the `rule!` macro
    let mut parser = rule!(
        CREATE ~ TABLE ~ #ident ~ ^"(" ~ (#ident ~ #ident ~ ","?)* ~ ")" ~ ";"
        : "CREATE TABLE statement"
    );
}
```

## Syntax

The `rule!` macro follows these rules:

| **Syntax**            | **Description**                                                 | **Expanded to**                         | **Operator Precedence** |
|-----------------------|-----------------------------------------------------------------|-----------------------------------------|-------------------------|
| `TOKEN`               | Matches a token by kind.                                        | `match_token(TOKEN)`                    | -                       |
| `"("`                 | Matches a token by its text.                                    | `match_text("(")`                       | -                       |
| `#fn_name`            | Calls an external nom parser function `fn_name`.                | `fn_name`                               | -                       |
| `#fn_name(a, b, c)`   | Calls an external nom parser function `fn_name` with arguments. | `fn_name(a, b, c)`                      | -                       |
| `a ~ b ~ c`           | Sequences parsers `a`, `b`, and `c`.                            | `nom::sequence::tuple((a, b, c))`       | 3 (Left Associative)    |
| `a+`                  | One or more repetitions.                                        | `nom::multi::many1(a)`                  | 4 (Postfix)             |
| `a*`                  | Zero or more repetitions.                                       | `nom::multi::many0(a)`                  | 4 (Postfix)             |
| `a?`                  | Optional parser.                                                | `nom::combinator::opt(a)`               | 4 (Postfix)             |
| `a \| b \| c`         | Choice between parsers `a`, `b`, and `c`.                       | `nom::branch::alt((a, b, c))`           | 1 (Left Associative)    |
| `&a`                  | Peeks at parser `a` without consuming input.                    | `nom::combinator::peek(a)`              | 5 (Prefix)              |
| `!a`                  | Negative lookahead for parser `a`.                              | `nom::combinator::not(a)`               | 5 (Prefix)              |
| `^a`                  | Cuts the parser `a`.                                            | `nom::combinator::cut(a)`               | 5 (Prefix)              |
| `... : "description"` | Adds a context description for error reporting.                 | `nom::error::context("description", a)` | 2 (Postfix)             |

## Example

See the [`sqlparser.rs`](examples/sqlparser.rs) file for a complete example parsing a simplified SQL `CREATE TABLE` statement.
