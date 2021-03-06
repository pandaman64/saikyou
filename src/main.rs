pub mod alpha;
pub mod ast;
pub mod parser;
pub mod typing;

fn main() -> anyhow::Result<()> {
    let program = r#"fun f x = x + 10.
fun g x y = 
    let x = f x;
    y + x.
fun h x = g x x.
fun const x y = x.
fun zero f x = x.
fun one f x = f x.
fun two f x = f (f x).
fun plus m n f x =
    let guide = f x;
    m f (n f x).
fun three f x = plus two one f x.
"#;

    let (_, functions) = parser::program(program)?;
    for f in functions.iter() {
        println!("{:?}", f);
    }

    let functions = {
        let mut env = alpha::Environment::new();
        let mut ctx = alpha::AlphaContext::new();

        functions
            .into_iter()
            .map(|f| {
                let name = f.name;
                let f = ctx.rename_func(&mut env, f)?;
                env.insert(name, f.name);
                Ok(f)
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    for f in functions.iter() {
        println!("{:?}", f);
    }

    let mut env = typing::Environment::new();
    let mut ctx = typing::TypeContext::new();
    let _functions = functions
        .into_iter()
        .map(|f| {
            let name = f.name;
            let f = ctx.infer_func(&mut env, f)?;
            env.insert(name, f.ext);

            println!("{}", f.pprint(&ctx));
            Ok(f)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    println!("ctx: {:?}", ctx);
    println!("env: {:?}", env);

    Ok(())
}
