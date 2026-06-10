//! Compose [`nom`] parsers with grammar-shaped expressions.
//!
//! [`rule!`] is a procedural macro. It does not tokenize input and does not
//! introduce a new parsing runtime. It expands a rule into ordinary `nom`
//! parser values, using terminal matcher functions supplied by the calling
//! crate.
//!
//! The generated code refers to `nom::...`, so crates using this macro must
//! depend on `nom` directly.
//!
//! # Quick Start
//!
//! ```
//! use nom::bytes::complete::tag;
//! use nom::character::complete::{alpha1, digit1};
//! use nom::combinator::map_res;
//! use nom::error::ParseError;
//! use nom::{IResult, Parser};
//! use nom_rule::rule;
//!
//! fn match_text<'a, Error>(
//!     text: &'static str,
//! ) -> impl Parser<&'a str, Output = &'a str, Error = Error>
//! where
//!     Error: ParseError<&'a str>,
//! {
//!     tag(text)
//! }
//!
//! fn number(input: &str) -> IResult<&str, u32> {
//!     map_res(digit1, str::parse).parse(input)
//! }
//!
//! fn assignment(input: &str) -> IResult<&str, (&str, u32)> {
//!     let mut parser = rule! {
//!         "let " ~ #alpha1 ~ " = " ~ #number ~ ";"
//!     };
//!
//!     let (rest, (_, name, _, value, _)) = parser.parse(input)?;
//!     Ok((rest, (name, value)))
//! }
//!
//! let (rest, (name, value)) = assignment("let answer = 42;").unwrap();
//!
//! assert_eq!(rest, "");
//! assert_eq!(name, "answer");
//! assert_eq!(value, 42);
//! ```
//!
//! # Terminals
//!
//! Terminals are the boundary between the grammar expression and your input
//! representation.
//!
//! | Rule form | Expanded call |
//! |-----------|---------------|
//! | `"text"` | `match_text("text")` |
//! | `TOKEN` or `path::TOKEN` | `match_token(TOKEN)` or `match_token(path::TOKEN)` |
//!
//! The matcher signatures are not fixed. They only need to return values that
//! implement [`nom::Parser`] for your input type.
//!
//! Custom terminal matcher paths can be supplied before the rule:
//!
//! ```ignore
//! let parser = rule! {
//!     crate::tokens::text, crate::tokens::kind,
//!     TokenKind::Create ~ #ident
//! };
//! ```
//!
//! # Rule Syntax
//!
//! | Form | Meaning | Output shape |
//! |------|---------|--------------|
//! | `"text"` | Match text with `match_text` | matcher output |
//! | `TOKEN` | Match a token with `match_token` | matcher output |
//! | `#parser` | Use an existing parser | parser output |
//! | `#parser(args...)` | Call a parser factory | parser output |
//! | `a ~ b` | Run parsers in sequence | `(A, B)` |
//! | `a \| b` | Try alternatives with [`nom::branch::alt`] | branch output |
//! | `a?` | Optional parser | `Option<A>` |
//! | `a*` | Zero or more matches | `Vec<A>` |
//! | `a+` | One or more matches | `Vec<A>` |
//! | `&a` | Peek without consuming input | `A` |
//! | `!a` | Negative lookahead | `()` |
//! | `^a` | Cut on failure | `A` |
//! | `a : "context"` | Add [`nom::error::context`] | `A` |
//!
//! Use parentheses to group subrules.
//!
//! Operator precedence, from low to high:
//!
//! 1. choice: `a | b`
//! 2. context: `a : "context"`
//! 3. sequence: `a ~ b`
//! 4. postfix operators: `?`, `*`, `+`
//! 5. prefix operators: `&`, `!`, `^`
//!
//! Choice branches must return compatible output shapes. An optional branch is
//! rejected before later alternatives because it can succeed without consuming
//! input.

use nom::branch::alt;
use nom::combinator::map;
use nom::combinator::opt;
use nom::error::make_error;
use nom::error::ErrorKind;
use nom::multi::many0;
use nom::Parser;
use nom::{IResult, Needed};
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
use quote::quote;
use quote::ToTokens;
use quote::TokenStreamExt;
use std::iter::{Cloned, Enumerate};
use std::ops::Deref;
use std::slice::Iter;
use syn::punctuated::Punctuated;
use syn::Token;

/// Expands a grammar-shaped expression into a `nom` parser.
///
/// See the [crate-level documentation](index.html) for terminal matcher
/// expectations, syntax, precedence, and output shapes.
#[proc_macro]
pub fn rule(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match expand_rule(tokens.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_rule(tokens: TokenStream) -> syn::Result<TokenStream> {
    let i: Vec<TokenTree> = tokens.into_iter().collect();

    // Attempt to parse the paths for match_text and match_token
    let (i, terminals) = if let Ok((rest, (match_text, _, match_token, _))) =
        (path, match_punct(','), path, match_punct(',')).parse(Input(&i))
    {
        (
            rest,
            Some(CustomTerminal {
                match_text: match_text.1,
                match_token: match_token.1,
            }),
        )
    } else {
        (Input(i.as_slice()), None)
    };

    let terminal = terminals.unwrap_or_else(|| CustomTerminal {
        match_text: Path {
            segments: vec![Ident::new("match_text", Span::call_site())],
        },
        match_token: Path {
            segments: vec![Ident::new("match_token", Span::call_site())],
        },
    });

    let rule = parse_rule(i.iter().cloned().collect())?;
    rule.check_return_type()?;
    Ok(rule.to_token_stream(&terminal))
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
    SubRule(Group),
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

#[derive(Debug, Clone)]
struct Input<'a>(&'a [TokenTree]);

impl Deref for Input<'_> {
    type Target = [TokenTree];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> nom::Input for Input<'a> {
    type Item = TokenTree;
    type Iter = Cloned<Iter<'a, TokenTree>>;
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
        self.iter().position(|b| predicate(b.clone()))
    }

    fn iter_elements(&self) -> Self::Iter {
        self.0.iter().cloned()
    }

    fn iter_indices(&self) -> Self::IterIndices {
        self.iter_elements().enumerate()
    }

    fn slice_index(&self, count: usize) -> Result<usize, Needed> {
        if self.len() >= count {
            Ok(count)
        } else {
            Err(Needed::new(count - self.len()))
        }
    }
}

fn match_punct<'a>(punct: char) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, TokenTree> {
    move |i| match i.first().and_then(|token| match token {
        TokenTree::Punct(p) if p.as_char() == punct => Some(token.clone()),
        _ => None,
    }) {
        Some(token) => Ok((Input(&i.0[1..]), token)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn group(i: Input) -> IResult<Input, Group> {
    match i.first().and_then(|token| match token {
        TokenTree::Group(group) => Some(group.clone()),
        _ => None,
    }) {
        Some(group) => Ok((Input(&i.0[1..]), group)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn literal(i: Input) -> IResult<Input, Literal> {
    match i.first().and_then(|token| match token {
        TokenTree::Literal(lit) => Some(lit.clone()),
        _ => None,
    }) {
        Some(lit) => Ok((Input(&i.0[1..]), lit)),
        _ => Err(nom::Err::Error(make_error(i, ErrorKind::Satisfy))),
    }
}

fn ident(i: Input) -> IResult<Input, Ident> {
    match i.first().and_then(|token| match token {
        TokenTree::Ident(ident) => Some(ident.clone()),
        _ => None,
    }) {
        Some(ident) => Ok((Input(&i.0[1..]), ident)),
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

fn parse_rule(tokens: TokenStream) -> syn::Result<Rule> {
    let i: Vec<TokenTree> = tokens.into_iter().collect();

    let (i, elems) = many0(parse_rule_element)
        .parse(Input(&i))
        .map_err(nom_error_to_syn)?;
    if !i.is_empty() {
        let rest: TokenStream = i.iter().cloned().collect();
        return Err(syn::Error::new_spanned(
            rest,
            "unable to parse the following rules",
        ));
    }

    let mut iter = elems.into_iter().peekable();
    let rule = RuleParser.parse(&mut iter).map_err(pratt_error_to_syn)?;
    if iter.peek().is_some() {
        let rest: Vec<_> = iter.collect();
        return Err(syn::Error::new(
            rest[0].span,
            format!("unable to parse the following rules: {rest:?}"),
        ));
    }

    Ok(rule)
}

fn nom_error_to_syn(error: nom::Err<nom::error::Error<Input<'_>>>) -> syn::Error {
    match error {
        nom::Err::Error(error) | nom::Err::Failure(error) => {
            let tokens: TokenStream = error.input.iter().cloned().collect();
            if tokens.is_empty() {
                syn::Error::new(Span::call_site(), "unable to parse rule")
            } else {
                syn::Error::new_spanned(tokens, "unable to parse rule")
            }
        }
        nom::Err::Incomplete(_) => syn::Error::new(Span::call_site(), "incomplete rule"),
    }
}

fn pratt_error_to_syn(error: PrattError<WithSpan, syn::Error>) -> syn::Error {
    match error {
        PrattError::UserError(error) => error,
        PrattError::EmptyInput => {
            syn::Error::new(Span::call_site(), "expected more tokens for rule")
        }
        PrattError::UnexpectedNilfix(input) => {
            syn::Error::new(input.span, "unable to parse the value")
        }
        PrattError::UnexpectedPrefix(input) => {
            syn::Error::new(input.span, "unable to parse the prefix operator")
        }
        PrattError::UnexpectedInfix(input) => {
            syn::Error::new(input.span, "unable to parse the binary operator")
        }
        PrattError::UnexpectedPostfix(input) => {
            syn::Error::new(input.span, "unable to parse the postfix operator")
        }
    }
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
            elem: RuleElement::SubRule(group),
        }),
        function_call,
        context,
    ))
    .parse(i)
}

struct RuleParser;

impl<I: Iterator<Item = WithSpan>> PrattParser<I> for RuleParser {
    type Error = syn::Error;
    type Input = WithSpan;
    type Output = Rule;

    fn query(&mut self, elem: &WithSpan) -> Result<Affix, syn::Error> {
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

    fn primary(&mut self, elem: WithSpan) -> Result<Rule, syn::Error> {
        let rule = match elem.elem {
            RuleElement::SubRule(group) => {
                if group.stream().is_empty() {
                    return Err(syn::Error::new(
                        group.span(),
                        "expected more tokens for rule",
                    ));
                }
                parse_rule(group.stream())?
            }
            RuleElement::MatchText(text) => Rule::MatchText(elem.span, text),
            RuleElement::MatchToken(token) => Rule::MatchToken(elem.span, token),
            RuleElement::ExternalFunction(func, args) => {
                Rule::ExternalFunction(elem.span, func, args)
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn infix(&mut self, lhs: Rule, elem: WithSpan, rhs: Rule) -> Result<Rule, syn::Error> {
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

    fn prefix(&mut self, elem: WithSpan, rhs: Rule) -> Result<Rule, syn::Error> {
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

    fn postfix(&mut self, lhs: Rule, elem: WithSpan) -> Result<Rule, syn::Error> {
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
    fn check_return_type(&self) -> syn::Result<ReturnType> {
        let ty = match self {
            Rule::MatchText(_, _) | Rule::MatchToken(_, _) | Rule::ExternalFunction(_, _, _) => {
                ReturnType::Unknown
            }
            Rule::Context(_, _, rule) | Rule::Peek(_, rule) => rule.check_return_type()?,
            Rule::Not(_, _) => ReturnType::Unit,
            Rule::Optional(_, rule) => ReturnType::Option(Box::new(rule.check_return_type()?)),
            Rule::Cut(_, rule) => rule.check_return_type()?,
            Rule::Many0(_, rule) | Rule::Many1(_, rule) => {
                ReturnType::Vec(Box::new(rule.check_return_type()?))
            }
            Rule::Sequence(_, rules) => {
                for rule in rules {
                    rule.check_return_type()?;
                }
                ReturnType::Vec(Box::new(ReturnType::Unknown))
            }
            Rule::Choice(_, rules) => {
                for slice in rules.windows(2) {
                    match (slice[0].check_return_type()?, slice[1].check_return_type()?) {
                        (ReturnType::Option(_), _) => {
                            return Err(syn::Error::new(
                                slice[0].span(),
                                "optional shouldn't be in a choice because it will shortcut the following branches",
                            ));
                        }
                        (a, b) if a != b => {
                            return Err(syn::Error::new(
                                slice[0]
                                    .span()
                                    .join(slice[1].span())
                                    .unwrap_or(Span::call_site()),
                                format!("type mismatched between {a:} and {b:}"),
                            ));
                        }
                        _ => (),
                    }
                }
                ReturnType::Vec(Box::new(rules[0].check_return_type()?))
            }
        };

        Ok(ty)
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
                quote! { (#list) }
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
