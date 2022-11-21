//! This crate simulates a 3x3x3 Rubik's cube by keeping track of the permutaion and orientation of edge and corner
//! cubies of the Rubiks Cube.
//!
//! Based on [The Fundamental Theorem of Cubology]
//!
//!
//!
//!
//!
//! [The Fundamental Theorem of Cubology]: http://www.sfu.ca/~jtmulhol/math302/puzzles-rc-cubology.html

mod cube;
mod cubies;
mod errors;
mod moves;
mod orientation;
mod permutation;
// mod search;

extern crate strum;
#[macro_use]
extern crate strum_macros;

pub use cube::Cube;
pub use moves::{MetricKind, Turn};

#[cfg(feature = "python")]
mod pyo3;
