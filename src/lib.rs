//! A procedural macro crate for writing nom parsers using a grammar-like syntax.
//!
//! The `nom-rule` crate provides the `rule!` macro, which allows you to define parsers in a DSL
//! similar to grammar rules. This improves readability and maintainability.
//!
//! # Syntax
//!
//! The macro follows these rules:
//!
//! | **Syntax**            | **Description**                                                                  | **Expanded to**                         | **Operator Precedence**  |
//! |-----------------------|----------------------------------------------------------------------------------|-----------------------------------------|--------------------------|
//! | `TOKEN`               | Matches a token by kind.                                                         | `match_token(TOKEN)`                    | -                        |
//! | `"@"`                 | Matches a token by its text.                                                     | `match_text("@")`                       | -                        |
//! | `#fn_name`            | Calls an external nom parser function `fn_name`.                                 | `fn_name`                               | -                        |
//! | `#fn_name(a, b, c)`   | Calls an external nom parser function `fn_name` with arguments.                  | `fn_name(a, b, c)`                      | -                        |
//! | `a ~ b ~ c`           | Sequences parsers `a`, `b`, and `c`.                                             | `nom::sequence::tuple((a, b, c))`       | 3 (Left Associative)     |
//! | `a+`                  | One or more repetitions.                                                         | `nom::multi::many1(a)`                  | 4 (Postfix)              |
//! | `a*`                  | Zero or more repetitions.                                                        | `nom::multi::many0(a)`                  | 4 (Postfix)              |
//! | `a?`                  | Optional parser.                                                                 | `nom::combinator::opt(a)`               | 4 (Postfix)              |
//! | `a \| b \| c`         | Choice between parsers `a`, `b`, and `c`.                                        | `nom::branch::alt((a, b, c))`           | 1 (Left Associative)     |
//! | `&a`                  | Peeks at parser `a` without consuming input.                                     | `nom::combinator::peek(a)`              | 5 (Prefix)               |
//! | `!a`                  | Negative lookahead for parser `a`.                                               | `nom::combinator::not(a)`               | 5 (Prefix)               |
//! | `^a`                  | Cuts the parser `a`.                                                             | `nom::combinator::cut(a)`               | 5 (Prefix)               |
//! | `... : "description"` | Adds a context description for error reporting.                                  | `nom::error::context("description", a)` | 2 (Postfix)              |
//! # Example
//!
//! ```rust-example
//! use nom_rule::rule;
//! use nom::IResult;
//!
//! // Define your match functions
//! fn match_text<'a>(text: &'a str) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>> {
//!     // Implementation
//! }
//!
//! fn match_token<'a>(kind: TokenKind) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, &'a Token<'a>> {
//!     // Implementation
//! }
//!
//! fn ident<'a>(input: Input<'a>) -> IResult<Input<'a>, &str> {
//!     // Implementation
//! }
//!
//! // Use the `rule!` macro
//! let mut parser = rule!(
//!     CREATE ~ TABLE ~ #ident ~ ^"(" ~ (#ident ~ #ident ~ ","?)* ~ ")" ~ ";"
//!     : "CREATE TABLE statement"
//! );
//! ```

use nom::branch::alt;
use nom::combinator::map;
use nom::combinator::opt;
use nom::error::make_error;
use nom::error::ErrorKind;
use nom::multi::many0;
use nom::IResult;
use nom::Parser;
use pratt::Affix;
use pratt::Associativity;
use pratt::PrattError;
use pratt::PrattParser;
use pratt::Precedence;
use proc_macro2::Group;
use proc_macro2::Ident;
use proc_macro2::Literal;
use proc_macro2::Punct;
use proc_macro2::Spacing;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use proc_macro2::TokenTree;
use proc_macro_error2::abort;
use proc_macro_error2::abort_call_site;
use proc_macro_error2::proc_macro_error;
use quote::quote;
use quote::ToTokens;
use quote::TokenStreamExt;
use syn::punctuated::Punctuated;
use syn::Token;

/// A procedural macro for writing nom parsers using a grammar-like syntax.
///
/// See the [crate level documentation](index.html) for more information.
#[proc_macro]
#[proc_macro_error]
pub fn rule(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let tokens: TokenStream = tokens.into();
    let i: Vec<TokenTree> = tokens.into_iter().collect();

    // Attempt to parse the paths for match_text and match_token
    let (i, terminals) = if let Ok((rest, (match_text, _, match_token, _))) =
        (path, match_punct(','), path, match_punct(',')).parse(&i)
    {
        (
            rest,
            Some(CustomTerminal {
                match_text: match_text.1,
                match_token: match_token.1,
            }),
        )
    } else {
        (i.as_slice(), None)
    };

    let terminal = terminals.unwrap_or_else(|| CustomTerminal {
        match_text: Path {
            segments: vec![Ident::new("match_text", Span::call_site())],
        },
        match_token: Path {
            segments: vec![Ident::new("match_token", Span::call_site())],
        },
    });

    let rule = parse_rule(i.iter().cloned().collect());
    rule.check_return_type();
    rule.to_token_stream(&terminal).into()
}

#[derive(Debug, Clone)]
struct Path {
    segments: Vec<Ident>,
}

#[derive(Debug, Clone)]
enum Rule {
    MatchText(Span, Literal),
    MatchToken(Span, Path),
    ExternalFunction(Span, Path, Option<Group>),
    Context(Span, Literal, Box<Rule>),
    Peek(Span, Box<Rule>),
    Not(Span, Box<Rule>),
    Optional(Span, Box<Rule>),
    Cut(Span, Box<Rule>),
    Many0(Span, Box<Rule>),
    Many1(Span, Box<Rule>),
    Sequence(Span, Vec<Rule>),
    Choice(Span, Vec<Rule>),
}

#[derive(Debug, Clone)]
enum RuleElement {
    MatchText(Literal),
    MatchToken(Path),
    ExternalFunction(Path, Option<Group>),
    Context(Literal),
    Peek,
    Not,
    Optional,
    Cut,
    Many0,
    Many1,
    Sequence,
    Choice,
    SubRule(Rule),
}

#[derive(Debug, Clone)]
struct WithSpan {
    elem: RuleElement,
    span: Span,
}

#[derive(Debug, Clone)]
enum ReturnType {
    Option(Box<ReturnType>),
    Vec(Box<ReturnType>),
    Unit,
    Unknown,
}

struct CustomTerminal {
    match_text: Path,
    match_token: Path,
}

type Input<'a> = &'a [TokenTree];

fn match_punct<'a>(punct: char) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, TokenTree> {
    move |i| match i.get(0).and_then(|token| match token {
        TokenTree::Punct(p) if p.as_char() == punct => Some(token.clone()),
        _ => None,
    }) {
        Some(token) => Ok((&i[1..], token)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn group(i: Input) -> IResult<Input, Group> {
    match i.get(0).and_then(|token| match token {
        TokenTree::Group(group) => Some(group.clone()),
        _ => None,
    }) {
        Some(group) => Ok((&i[1..], group)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn literal(i: Input) -> IResult<Input, Literal> {
    match i.get(0).and_then(|token| match token {
        TokenTree::Literal(lit) => Some(lit.clone()),
        _ => None,
    }) {
        Some(lit) => Ok((&i[1..], lit)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn ident(i: Input) -> IResult<Input, Ident> {
    match i.get(0).and_then(|token| match token {
        TokenTree::Ident(ident) => Some(ident.clone()),
        _ => None,
    }) {
        Some(ident) => Ok((&i[1..], ident)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn path(i: Input) -> IResult<Input, (Span, Path)> {
    map(
        (ident, many0((match_punct(':'), match_punct(':'), ident))),
        |(head, tail)| {
            let mut segments = vec![head.clone()];
            segments.extend(tail.into_iter().map(|(_, _, segment)| segment));
            let span = segments
                .iter()
                .try_fold(head.span(), |span, seg| span.join(seg.span()))
                .unwrap_or(Span::call_site());
            (span, Path { segments })
        },
    )
    .parse(i)
}

fn parse_rule(tokens: TokenStream) -> Rule {
    let i: Vec<TokenTree> = tokens.into_iter().collect();

    let (i, elems) = many0(parse_rule_element).parse(&i).unwrap();
    if !i.is_empty() {
        let rest: TokenStream = i.iter().cloned().collect();
        abort!(rest, "unable to parse the following rules: {}", rest);
    }

    let mut iter = elems.into_iter().peekable();
    let rule = unwrap_pratt(RuleParser.parse(&mut iter));
    if iter.peek().is_some() {
        let rest: Vec<_> = iter.collect();
        abort!(
            rest[0].span,
            "unable to parse the following rules: {:?}",
            rest
        );
    }

    rule
}

fn parse_rule_element(i: Input) -> IResult<Input, WithSpan> {
    let function_call = |i| {
        let (i, hashtag) = match_punct('#')(i)?;
        let (i, (path_span, fn_path)) = path(i)?;
        let (i, args) = opt(group).parse(i)?;
        let span = hashtag.span().join(path_span).unwrap_or(Span::call_site());
        let span = args
            .as_ref()
            .and_then(|args| args.span().join(span))
            .unwrap_or(span);

        Ok((
            i,
            WithSpan {
                elem: RuleElement::ExternalFunction(fn_path, args),
                span,
            },
        ))
    };
    let context = map((match_punct(':'), literal), |(colon, msg)| {
        let span = colon.span().join(msg.span()).unwrap_or(Span::call_site());
        WithSpan {
            elem: RuleElement::Context(msg),
            span,
        }
    });
    alt((
        map(match_punct('|'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Choice,
        }),
        map(match_punct('*'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Many0,
        }),
        map(match_punct('+'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Many1,
        }),
        map(match_punct('?'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Optional,
        }),
        map(match_punct('^'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Cut,
        }),
        map(match_punct('&'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Peek,
        }),
        map(match_punct('!'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Not,
        }),
        map(match_punct('~'), |token| WithSpan {
            span: token.span(),
            elem: RuleElement::Sequence,
        }),
        map(literal, |lit| WithSpan {
            span: lit.span(),
            elem: RuleElement::MatchText(lit),
        }),
        map(path, |(span, p)| WithSpan {
            span,
            elem: RuleElement::MatchToken(p),
        }),
        map(group, |group| WithSpan {
            span: group.span(),
            elem: RuleElement::SubRule(parse_rule(group.stream())),
        }),
        function_call,
        context,
    ))
    .parse(i)
}

fn unwrap_pratt(res: Result<Rule, PrattError<WithSpan, pratt::NoError>>) -> Rule {
    match res {
        Ok(res) => res,
        Err(PrattError::EmptyInput) => abort_call_site!("expected more tokens for rule"),
        Err(PrattError::UnexpectedNilfix(input)) => {
            abort!(input.span, "unable to parse the value")
        }
        Err(PrattError::UnexpectedPrefix(input)) => {
            abort!(input.span, "unable to parse the prefix operator")
        }
        Err(PrattError::UnexpectedInfix(input)) => {
            abort!(input.span, "unable to parse the binary operator")
        }
        Err(PrattError::UnexpectedPostfix(input)) => {
            abort!(input.span, "unable to parse the postfix operator")
        }
        Err(PrattError::UserError(_)) => unreachable!(),
    }
}

struct RuleParser;

impl<I: Iterator<Item = WithSpan>> PrattParser<I> for RuleParser {
    type Error = pratt::NoError;
    type Input = WithSpan;
    type Output = Rule;

    fn query(&mut self, elem: &WithSpan) -> pratt::Result<Affix> {
        let affix = match elem.elem {
            RuleElement::Choice => Affix::Infix(Precedence(1), Associativity::Left),
            RuleElement::Context(_) => Affix::Postfix(Precedence(2)),
            RuleElement::Sequence => Affix::Infix(Precedence(3), Associativity::Left),
            RuleElement::Optional => Affix::Postfix(Precedence(4)),
            RuleElement::Many1 => Affix::Postfix(Precedence(4)),
            RuleElement::Many0 => Affix::Postfix(Precedence(4)),
            RuleElement::Cut => Affix::Prefix(Precedence(5)),
            RuleElement::Peek => Affix::Prefix(Precedence(5)),
            RuleElement::Not => Affix::Prefix(Precedence(5)),
            _ => Affix::Nilfix,
        };
        Ok(affix)
    }

    fn primary(&mut self, elem: WithSpan) -> pratt::Result<Rule> {
        let rule = match elem.elem {
            RuleElement::SubRule(rule) => rule,
            RuleElement::MatchText(text) => Rule::MatchText(elem.span, text),
            RuleElement::MatchToken(token) => Rule::MatchToken(elem.span, token),
            RuleElement::ExternalFunction(func, args) => {
                Rule::ExternalFunction(elem.span, func, args)
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn infix(&mut self, lhs: Rule, elem: WithSpan, rhs: Rule) -> pratt::Result<Rule> {
        let rule = match elem.elem {
            RuleElement::Sequence => match lhs {
                Rule::Sequence(span, mut seq) => {
                    let span = span
                        .join(elem.span)
                        .unwrap_or(Span::call_site())
                        .join(rhs.span())
                        .unwrap_or(Span::call_site());
                    seq.push(rhs);
                    Rule::Sequence(span, seq)
                }
                lhs => {
                    let span = lhs.span().join(rhs.span()).unwrap_or(Span::call_site());
                    Rule::Sequence(span, vec![lhs, rhs])
                }
            },
            RuleElement::Choice => match lhs {
                Rule::Choice(span, mut choices) => {
                    let span = span
                        .join(elem.span)
                        .unwrap_or(Span::call_site())
                        .join(rhs.span())
                        .unwrap_or(Span::call_site());
                    choices.push(rhs);
                    Rule::Choice(span, choices)
                }
                lhs => {
                    let span = lhs.span().join(rhs.span()).unwrap_or(Span::call_site());
                    Rule::Choice(span, vec![lhs, rhs])
                }
            },
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn prefix(&mut self, elem: WithSpan, rhs: Rule) -> pratt::Result<Rule> {
        let rule = match elem.elem {
            RuleElement::Cut => {
                let span = elem.span.join(rhs.span()).unwrap_or(Span::call_site());
                Rule::Cut(span, Box::new(rhs))
            }
            RuleElement::Peek => {
                let span = elem.span.join(rhs.span()).unwrap_or(Span::call_site());
                Rule::Peek(span, Box::new(rhs))
            }
            RuleElement::Not => {
                let span = elem.span.join(rhs.span()).unwrap_or(Span::call_site());
                Rule::Not(span, Box::new(rhs))
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn postfix(&mut self, lhs: Rule, elem: WithSpan) -> pratt::Result<Rule> {
        let rule = match elem.elem {
            RuleElement::Optional => {
                let span = lhs.span().join(elem.span).unwrap_or(Span::call_site());
                Rule::Optional(span, Box::new(lhs))
            }
            RuleElement::Many0 => {
                let span = lhs.span().join(elem.span).unwrap_or(Span::call_site());
                Rule::Many0(span, Box::new(lhs))
            }
            RuleElement::Many1 => {
                let span = lhs.span().join(elem.span).unwrap_or(Span::call_site());
                Rule::Many1(span, Box::new(lhs))
            }
            RuleElement::Context(msg) => {
                let span = lhs.span().join(elem.span).unwrap_or(Span::call_site());
                Rule::Context(span, msg, Box::new(lhs))
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }
}

impl std::fmt::Display for ReturnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReturnType::Option(ty) => write!(f, "Option<{}>", ty),
            ReturnType::Vec(ty) => write!(f, "Vec<{}>", ty),
            ReturnType::Unit => write!(f, "()"),
            ReturnType::Unknown => write!(f, "_"),
        }
    }
}

impl PartialEq for ReturnType {
    fn eq(&self, other: &ReturnType) -> bool {
        match (self, other) {
            (ReturnType::Option(lhs), ReturnType::Option(rhs)) => lhs == rhs,
            (ReturnType::Vec(lhs), ReturnType::Vec(rhs)) => lhs == rhs,
            (ReturnType::Unit, ReturnType::Unit) => true,
            (ReturnType::Unknown, _) => true,
            (_, ReturnType::Unknown) => true,
            _ => false,
        }
    }
}

impl Rule {
    fn check_return_type(&self) -> ReturnType {
        match self {
            Rule::MatchText(_, _) | Rule::MatchToken(_, _) | Rule::ExternalFunction(_, _, _) => {
                ReturnType::Unknown
            }
            Rule::Context(_, _, rule) | Rule::Peek(_, rule) => rule.check_return_type(),
            Rule::Not(_, _) => ReturnType::Unit,
            Rule::Optional(_, rule) => ReturnType::Option(Box::new(rule.check_return_type())),
            Rule::Cut(_, rule) => rule.check_return_type(),
            Rule::Many0(_, rule) | Rule::Many1(_, rule) => {
                ReturnType::Vec(Box::new(rule.check_return_type()))
            }
            Rule::Sequence(_, rules) => {
                rules.iter().for_each(|rule| {
                    rule.check_return_type();
                });
                ReturnType::Vec(Box::new(ReturnType::Unknown))
            }
            Rule::Choice(_, rules) => {
                for slice in rules.windows(2) {
                    match (slice[0].check_return_type(), slice[1].check_return_type()) {
                        (ReturnType::Option(_), _) => {
                            abort!(
                                slice[0].span(),
                                "optional shouldn't be in a choice because it will shortcut the following branches",
                            )
                        }
                        (a, b) if a != b => abort!(
                            slice[0]
                                .span()
                                .join(slice[1].span())
                                .unwrap_or(Span::call_site()),
                            "type mismatched between {:} and {:}",
                            a,
                            b,
                        ),
                        _ => (),
                    }
                }
                ReturnType::Vec(Box::new(rules[0].check_return_type()))
            }
        }
    }

    fn span(&self) -> Span {
        match self {
            Rule::MatchText(span, _)
            | Rule::MatchToken(span, _)
            | Rule::ExternalFunction(span, _, _)
            | Rule::Context(span, _, _)
            | Rule::Peek(span, _)
            | Rule::Not(span, _)
            | Rule::Optional(span, _)
            | Rule::Cut(span, _)
            | Rule::Many0(span, _)
            | Rule::Many1(span, _)
            | Rule::Sequence(span, _)
            | Rule::Choice(span, _) => *span,
        }
    }

    fn to_tokens(&self, terminal: &CustomTerminal, tokens: &mut TokenStream) {
        let token = match self {
            Rule::MatchText(_, text) => {
                let match_text = &terminal.match_text;
                quote! { #match_text (#text) }
            }
            Rule::MatchToken(_, token) => {
                let match_token = &terminal.match_token;
                quote! { #match_token (#token) }
            }
            Rule::ExternalFunction(_, name, arg) => {
                quote! { #name #arg }
            }
            Rule::Context(_, msg, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::error::context(#msg, #rule) }
            }
            Rule::Peek(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::peek(#rule) }
            }
            Rule::Not(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::not(#rule) }
            }
            Rule::Optional(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::opt(#rule) }
            }
            Rule::Cut(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::cut(#rule) }
            }
            Rule::Many0(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::multi::many0(#rule) }
            }
            Rule::Many1(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::multi::many1(#rule) }
            }
            Rule::Sequence(_, rules) => {
                let list: Punctuated<TokenStream, Token![,]> = rules
                    .iter()
                    .map(|rule| rule.to_token_stream(terminal))
                    .collect();
                quote! { nom::sequence::tuple((#list)) }
            }
            Rule::Choice(_, rules) => {
                let list: Punctuated<TokenStream, Token![,]> = rules
                    .iter()
                    .map(|rule| rule.to_token_stream(terminal))
                    .collect();
                quote! { nom::branch::alt((#list)) }
            }
        };

        tokens.extend(token);
    }

    fn to_token_stream(&self, terminal: &CustomTerminal) -> TokenStream {
        let mut tokens = TokenStream::new();
        self.to_tokens(terminal, &mut tokens);
        tokens
    }
}

impl ToTokens for Path {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                // Double colon `::`
                tokens.append(Punct::new(':', Spacing::Joint));
                tokens.append(Punct::new(':', Spacing::Alone));
            }
            segment.to_tokens(tokens);
        }
    }
}
