use uiua_parser::parse::parse;
use uiua_parser::{InputSrc, Inputs};

fn main() {
    let input = "⇡10";
    let mut inputs = Inputs::default();
    let src = InputSrc::Literal(input.into());
    let (items, errors, diagnostics) = parse(input, src, &mut inputs);
    println!("Items: {items:?}");
    println!("Errors: {errors:?}");
    println!("Diagnostics: {diagnostics:?}");
}
