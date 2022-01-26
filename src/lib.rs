use pratt::{Affix, Associativity, PrattError, PrattParser, Precedence};
use proc_macro2::{Group, Ident, Literal, Span, TokenStream, TokenTree};
use proc_macro_error::{abort, abort_call_site, proc_macro_error};
use quote::quote;
use syn::{punctuated::Punctuated, Token};

#[proc_macro]
#[proc_macro_error]
pub fn rule(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let tokens: TokenStream = tokens.into();
    let mut iter = tokens.into_iter().peekable();

    let whitespace = match iter.next() {
        Some(TokenTree::Group(group)) => match iter.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => group,
            Some(tt) => abort!(tt, "Expected ',' after the whitespace parameter"),
            None => abort_call_site!("Expected ',' after the whitespace parameter"),
        },
        Some(tt) => abort!(
            tt,
            "Expected the first parameter to be a whitespace parser between parentheses, \n\n\
             help: consider 'rule!((nom::character::complete::multispace0), ...)'"
        ),
        None => abort_call_site!("Unexpected empty rule"),
    };

    let rule = unwrap_pratt(RuleParser.parse(&mut iter));

    if iter.peek().is_some() {
        let rest: TokenStream = iter.collect();
        abort!(rest, "Unable to parse the following rules: {}", rest);
    }

    rule.check_return_type();

    rule.to_token_stream(&whitespace).into()
}

fn unwrap_pratt(res: Result<Rule, PrattError<TokenTree, pratt::NoError>>) -> Rule {
    match res {
        Ok(res) => res,
        Err(PrattError::EmptyInput) => abort_call_site!("Unexpected empty rule"),
        Err(PrattError::UnexpectedNilfix(input)) => {
            abort!(input.span(), "Unable to parse the value")
        }
        Err(PrattError::UnexpectedPrefix(input)) => {
            abort!(input.span(), "Unable to parse the prefix operator")
        }
        Err(PrattError::UnexpectedInfix(input)) => {
            abort!(input.span(), "Unable to parse the binary operator")
        }
        Err(PrattError::UnexpectedPostfix(input)) => {
            abort!(input.span(), "Unable to parse the postfix operator")
        }
        Err(PrattError::UserError(_)) => unreachable!(),
    }
}

struct RuleParser;

impl<I: Iterator<Item = TokenTree>> PrattParser<I> for RuleParser {
    type Error = pratt::NoError;
    type Input = TokenTree;
    type Output = Rule;

    fn query(&mut self, tree: &TokenTree) -> pratt::Result<Affix> {
        let affix = match tree {
            TokenTree::Punct(punct) if punct.as_char() == '|' => {
                Affix::Infix(Precedence(1), Associativity::Left)
            }
            TokenTree::Punct(punct) if punct.as_char() == '~' => {
                Affix::Infix(Precedence(2), Associativity::Left)
            }
            TokenTree::Punct(punct) if punct.as_char() == '?' => Affix::Postfix(Precedence(3)),
            TokenTree::Punct(punct) if punct.as_char() == '+' => Affix::Postfix(Precedence(3)),
            TokenTree::Punct(punct) if punct.as_char() == '*' => Affix::Postfix(Precedence(3)),
            TokenTree::Punct(punct) if punct.as_char() == '&' => Affix::Prefix(Precedence(4)),
            TokenTree::Punct(punct) if punct.as_char() == '!' => Affix::Prefix(Precedence(4)),
            TokenTree::Punct(punct) if punct.as_char() == '^' => Affix::Prefix(Precedence(5)),
            _ => Affix::Nilfix,
        };
        Ok(affix)
    }

    fn primary(&mut self, tree: TokenTree) -> pratt::Result<Rule> {
        let rule = match tree {
            TokenTree::Ident(ident) => Rule::ExternalFunction(ident),
            TokenTree::Literal(lit) => Rule::Tag(lit),
            TokenTree::Group(group) => {
                unwrap_pratt(RuleParser.parse(&mut group.stream().into_iter()))
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn infix(&mut self, lhs: Rule, tree: TokenTree, rhs: Rule) -> pratt::Result<Rule> {
        let rule = match tree.clone() {
            TokenTree::Punct(punct) if punct.as_char() == '~' => match lhs {
                Rule::Sequence(span, mut seq) => {
                    let span = span.join(tree.span()).unwrap().join(rhs.span()).unwrap();
                    seq.push(rhs);
                    Rule::Sequence(span, seq)
                }
                lhs => {
                    let span = lhs.span().join(rhs.span()).unwrap();
                    Rule::Sequence(span, vec![lhs, rhs])
                }
            },
            TokenTree::Punct(punct) if punct.as_char() == '|' => match lhs {
                Rule::Choice(span, mut choices) => {
                    let span = span.join(tree.span()).unwrap().join(rhs.span()).unwrap();
                    choices.push(rhs);
                    Rule::Choice(span, choices)
                }
                lhs => {
                    let span = lhs.span().join(rhs.span()).unwrap();
                    Rule::Choice(span, vec![lhs, rhs])
                }
            },
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn prefix(&mut self, tree: TokenTree, rhs: Rule) -> pratt::Result<Rule> {
        let rule = match tree.clone() {
            TokenTree::Punct(punct) if punct.as_char() == '^' => match rhs.clone() {
                Rule::Tag(tag) => {
                    let span = tree.span().join(rhs.span()).unwrap();
                    Rule::TagNoCase(span, tag)
                }
                _ => abort!(
                    tree,
                    "Symbol '^' is only allowed to be followed by a string literal"
                ),
            },
            TokenTree::Punct(punct) if punct.as_char() == '&' => {
                let span = tree.span().join(rhs.span()).unwrap();
                Rule::PositivePredicate(span, Box::new(rhs))
            }
            TokenTree::Punct(punct) if punct.as_char() == '!' => {
                let span = tree.span().join(rhs.span()).unwrap();
                Rule::NegativePredicate(span, Box::new(rhs))
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }

    fn postfix(&mut self, lhs: Rule, tree: TokenTree) -> pratt::Result<Rule> {
        let rule = match tree.clone() {
            TokenTree::Punct(punct) if punct.as_char() == '?' => {
                let span = lhs.span().join(tree.span()).unwrap();
                Rule::Optional(span, Box::new(lhs))
            }
            TokenTree::Punct(punct) if punct.as_char() == '*' => {
                let span = lhs.span().join(tree.span()).unwrap();
                Rule::Many0(span, Box::new(lhs))
            }
            TokenTree::Punct(punct) if punct.as_char() == '+' => {
                let span = lhs.span().join(tree.span()).unwrap();
                Rule::Many1(span, Box::new(lhs))
            }
            _ => unreachable!(),
        };
        Ok(rule)
    }
}

#[derive(Debug, Clone)]
enum Rule {
    Tag(Literal),
    TagNoCase(Span, Literal),
    ExternalFunction(Ident),
    PositivePredicate(Span, Box<Rule>),
    NegativePredicate(Span, Box<Rule>),
    Optional(Span, Box<Rule>),
    Many0(Span, Box<Rule>),
    Many1(Span, Box<Rule>),
    Sequence(Span, Vec<Rule>),
    Choice(Span, Vec<Rule>),
}

#[derive(Debug, Clone)]
enum ReturnType {
    Option(Box<ReturnType>),
    Vec(Box<ReturnType>),
    Str,
    Unit,
    Unknown,
}

impl std::fmt::Display for ReturnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReturnType::Option(ty) => write!(f, "Option<{}>", ty),
            ReturnType::Vec(ty) => write!(f, "Vec<{}>", ty),
            ReturnType::Str => write!(f, "&str"),
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
            (ReturnType::Str, ReturnType::Str) => true,
            (ReturnType::Unit, ReturnType::Unit) => true,
            (ReturnType::Unknown, _) => true,
            _ => false,
        }
    }
}

impl Rule {
    fn check_return_type(&self) -> ReturnType {
        match self {
            Rule::Tag(_) | Rule::TagNoCase(_, _) => ReturnType::Str,
            Rule::ExternalFunction(_) => ReturnType::Unknown,
            Rule::PositivePredicate(_, _) | Rule::NegativePredicate(_, _) => ReturnType::Unit,
            Rule::Optional(_, rule) => ReturnType::Option(Box::new(rule.check_return_type())),
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
                                "Optional shouldn't be in a choice because it will shortcut the following branches",
                            )
                        }
                        (a, b) if a != b => abort!(
                            slice[0].span().join(slice[1].span()).unwrap(),
                            "Type mismatched between {:} and {:}",
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
            Rule::Tag(lit) => lit.span(),
            Rule::ExternalFunction(ident) => ident.span(),
            Rule::TagNoCase(span, _)
            | Rule::PositivePredicate(span, _)
            | Rule::NegativePredicate(span, _)
            | Rule::Optional(span, _)
            | Rule::Many0(span, _)
            | Rule::Many1(span, _)
            | Rule::Sequence(span, _)
            | Rule::Choice(span, _) => *span,
        }
    }

    fn to_tokens(&self, whitespace: &Group, tokens: &mut TokenStream) {
        let token = match self {
            Rule::Tag(tag) => {
                quote! { nom::bytes::complete::tag(#tag) }
            }
            Rule::TagNoCase(_, tag) => {
                quote! { nom::bytes::complete::tag_no_case(#tag) }
            }
            Rule::ExternalFunction(ident) => {
                quote! { #ident }
            }
            Rule::PositivePredicate(_, rule) => {
                let rule = rule.to_token_stream(whitespace);
                quote! { nom::combinator::map(nom::combinator::peek(#rule), |_| ()) }
            }
            Rule::NegativePredicate(_, rule) => {
                let rule = rule.to_token_stream(whitespace);
                quote! { nom::combinator::not(#rule) }
            }
            Rule::Optional(_, rule) => {
                let rule = rule.to_token_stream(whitespace);
                quote! { nom::combinator::opt(#rule) }
            }
            Rule::Many0(_, rule) => {
                let rule = rule.to_token_stream(whitespace);
                quote! { nom::multi::many0(nom::sequence::preceded(#whitespace, #rule)) }
            }
            Rule::Many1(_, rule) => {
                let rule = rule.to_token_stream(whitespace);
                quote! { nom::multi::many1(nom::sequence::preceded(#whitespace, #rule)) }
            }
            Rule::Sequence(_, rules) => {
                let list: Punctuated<TokenStream, Token![,]> = rules
                    .iter()
                    .map(|rule| {
                        let rule = rule.to_token_stream(whitespace);
                        quote! { nom::sequence::preceded(#whitespace, #rule) }
                    })
                    .collect();
                quote! { nom::sequence::tuple((#list)) }
            }
            Rule::Choice(_, rules) => {
                let list: Punctuated<TokenStream, Token![,]> = rules
                    .iter()
                    .map(|rule| rule.to_token_stream(whitespace))
                    .collect();
                quote! { nom::branch::alt((#list)) }
            }
        };

        tokens.extend(token);
    }

    fn to_token_stream(&self, whitespace: &Group) -> TokenStream {
        let mut tokens = TokenStream::new();
        self.to_tokens(whitespace, &mut tokens);
        tokens
    }
}
