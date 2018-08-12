use std::env;
use std::process::Command;

fn main() {
    let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // set path so it can find cuda libraries.
    println!("cargo:rustc-link-search={}", project_dir);
    let mut cuda_path = match env::var("CUDA_PATH") {
        Ok(path) => {
            let path = if path.chars().last().unwrap() != '/' {
                path + &"/"
            } else {
                path
            };
            Ok(path)
        }
        Err(_) => Err(()),
    }.unwrap().to_owned();

    cuda_path.push_str("/lib64");
    let mut search_string = "cargo:rustc-link-search=native=".to_owned();

    search_string.push_str(&cuda_path);
    println!("{}", search_string);
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let status = Command::new("make")
        .arg("clean")
        .arg("-C")
        .arg(format!("{}/weld_rt/cpp/", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

    let status = Command::new("make")
        .arg("-C")
        .arg(format!("{}/weld_rt/cpp/", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

    // Link C++ standard library and some Mac-specific libraries
    let target = env::var("TARGET").unwrap();
    if target == "x86_64-apple-darwin" {
        let libs = vec!["z", "c++"];
        for lib in libs {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Link the weldrt C++ library
    println!("cargo:rustc-link-lib=static=weldrt");
    println!("cargo:rustc-link-search=native={}/weld_rt/cpp", project_dir);
}
