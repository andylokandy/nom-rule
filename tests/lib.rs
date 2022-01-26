use nom::{bytes::complete::take_while1, character::is_alphanumeric, IResult};

macro_rules! rule {
    ($($tt:tt)*) => { nom_rule::rule!((nom::character::complete::multispace0), $($tt)*) }
}

#[test]
fn sql_create_table() {
    let mut rule = rule!(
        ^"CREATE" ~ ^"TABLE" ~ ident ~ "(" ~ (ident ~ ident ~ ","?)* ~ ")" ~ ";"
    );

    assert_eq!(
        rule("create table user (id int, name varchar);"),
        Ok((
            "",
            (
                "create",
                "table",
                "user",
                "(",
                vec![("id", "int", Some(",")), ("name", "varchar", None)],
                ")",
                ";"
            )
        ))
    );

    assert_eq!(
        rule("create table user ();"),
        Ok(("", ("create", "table", "user", "(", vec![], ")", ";")))
    );
}

pub fn ident(i: &str) -> IResult<&str, &str> {
    take_while1(is_sql_identifier)(i)
}

pub fn is_sql_identifier(chr: char) -> bool {
    is_alphanumeric(chr as u8) || chr == '_' || chr == '@'
}
