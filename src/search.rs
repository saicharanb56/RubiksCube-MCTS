use std::{
    hash::{Hash, Hasher},
    rc::Rc,
    sync::{Arc, Mutex},
};

struct Node {
    state: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
    repr: Vec<bool>,
    is_solved: bool,
    cost_to_go: f64,
    from_state: Option<Mutex<Arc<Node>>>,
    traj: Vec<Rc<Node>>,
}

struct GBFS {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.repr.hash(state);
    }
}
