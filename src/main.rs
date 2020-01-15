pub mod ast;
pub mod parser;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error + 'static>> {
    let program = r#"
fun f x = x + 10.
fun g x y = 
    let z = f x;
    y + z.
fun h x = g x x.
"#;

    let (_, functions) = parser::program(program)?;
    for f in functions.iter() {
        println!("{:?}", f);
    }

    Ok(())
}
