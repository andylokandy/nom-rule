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

    let match_text = match iter.next() {
        Some(TokenTree::Group(group)) => match iter.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => group,
            Some(tt) => abort!(tt, "expected ',' after the match_text parameter"),
            None => abort_call_site!("expected ',' after the match_text parameter"),
        },
        Some(tt) => abort!(
            tt,
            "expected the first parameter to be a match_text parser between parentheses"
        ),
        None => abort_call_site!("unexpected empty match_text"),
    };
    let match_token = match iter.next() {
        Some(TokenTree::Group(group)) => match iter.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => group,
            Some(tt) => abort!(tt, "expected ',' after the match_token parameter"),
            None => abort_call_site!("expected ',' after the match_token parameter"),
        },
        Some(tt) => abort!(
            tt,
            "expected the second parameter to be a match_token parser between parentheses"
        ),
        None => abort_call_site!("unexpected empty match_token"),
    };

    let rule = unwrap_pratt(RuleParser.parse(&mut iter));

    if iter.peek().is_some() {
        let rest: TokenStream = iter.collect();
        abort!(rest, "unable to parse the following rules: {}", rest);
    }

    rule.check_return_type();

    let terminal = CustomTerminal {
        match_text,
        match_token,
    };
    rule.to_token_stream(&terminal).into()
}

fn unwrap_pratt(res: Result<Rule, PrattError<TokenTree, pratt::NoError>>) -> Rule {
    match res {
        Ok(res) => res,
        Err(PrattError::EmptyInput) => abort_call_site!("unexpected empty rule"),
        Err(PrattError::UnexpectedNilfix(input)) => {
            abort!(input.span(), "unable to parse the value")
        }
        Err(PrattError::UnexpectedPrefix(input)) => {
            abort!(input.span(), "unable to parse the prefix operator")
        }
        Err(PrattError::UnexpectedInfix(input)) => {
            abort!(input.span(), "unable to parse the binary operator")
        }
        Err(PrattError::UnexpectedPostfix(input)) => {
            abort!(input.span(), "unable to parse the postfix operator")
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
            TokenTree::Punct(punct) if punct.as_char() == '#' => Affix::Prefix(Precedence(5)),
            _ => Affix::Nilfix,
        };
        Ok(affix)
    }

    fn primary(&mut self, tree: TokenTree) -> pratt::Result<Rule> {
        let rule = match tree {
            TokenTree::Ident(ident) => Rule::MatchToken(ident),
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
            TokenTree::Punct(punct) if punct.as_char() == '#' => match rhs.clone() {
                Rule::MatchToken(ident) => {
                    let span = tree.span().join(rhs.span()).unwrap();
                    Rule::ExternalFunction(span, ident)
                }
                _ => abort!(
                    tree,
                    "symbol '#' is only allowed to be followed by an function identifier"
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

#[derive(Clone)]
enum Rule {
    Tag(Literal),
    MatchToken(Ident),
    ExternalFunction(Span, Ident),
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
    Unit,
    Unknown,
}

struct CustomTerminal {
    match_text: Group,
    match_token: Group,
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
            Rule::Tag(_) | Rule::MatchToken(_) | Rule::ExternalFunction(_, _) => {
                ReturnType::Unknown
            }
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
                                "optional shouldn't be in a choice because it will shortcut the following branches",
                            )
                        }
                        (a, b) if a != b => abort!(
                            slice[0].span().join(slice[1].span()).unwrap(),
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
            Rule::Tag(lit) => lit.span(),
            Rule::MatchToken(ident) => ident.span(),
            Rule::ExternalFunction(span, _)
            | Rule::PositivePredicate(span, _)
            | Rule::NegativePredicate(span, _)
            | Rule::Optional(span, _)
            | Rule::Many0(span, _)
            | Rule::Many1(span, _)
            | Rule::Sequence(span, _)
            | Rule::Choice(span, _) => *span,
        }
    }

    fn to_tokens(&self, terminal: &CustomTerminal, tokens: &mut TokenStream) {
        let token = match self {
            Rule::Tag(tag) => {
                let match_text = &terminal.match_text;
                quote! { #match_text (#tag) }
            }
            Rule::MatchToken(token) => {
                let match_token = &terminal.match_token;
                quote! { #match_token (#token) }
            }
            Rule::ExternalFunction(_, ident) => {
                quote! { #ident }
            }
            Rule::PositivePredicate(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::map(nom::combinator::peek(#rule), |_| ()) }
            }
            Rule::NegativePredicate(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::not(#rule) }
            }
            Rule::Optional(_, rule) => {
                let rule = rule.to_token_stream(terminal);
                quote! { nom::combinator::opt(#rule) }
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
