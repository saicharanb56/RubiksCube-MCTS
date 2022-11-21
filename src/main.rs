use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

fn main() {
    let v: Vec<bool> = vec![false; 480];
    let mut hasher = DefaultHasher::new();

    v.hash(&mut hasher);
    println!("hash : {}", hasher.finish());

    v.hash(&mut hasher);
    println!("hash : {}", hasher.finish());
}
