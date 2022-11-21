use crate::{
    cubies::{NUM_CORNERS, NUM_EDGES},
    errors::CubeError,
};

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Permutation {
    permutation: Vec<u8>,
}

impl Permutation {
    pub fn edge() -> Permutation {
        Permutation {
            permutation: (0..NUM_EDGES).collect(),
        }
    }

    pub fn corner() -> Permutation {
        Permutation {
            permutation: (0..NUM_CORNERS).collect(),
        }
    }

    pub fn new_with_permutation(perm: &[u8]) -> Permutation {
        Permutation {
            permutation: perm.to_vec(),
        }
    }

    pub fn swap_four_cubies(&mut self, cubicle_a: u8, cubicle_b: u8, cubicle_c: u8, cubicle_d: u8) {
        self.permutation
            .swap(cubicle_a as usize, cubicle_b as usize);
        self.permutation
            .swap(cubicle_a as usize, cubicle_c as usize);
        self.permutation
            .swap(cubicle_a as usize, cubicle_d as usize);
    }

    pub fn swap_two_cubies(&mut self, cubicle_a: u8, cubicle_b: u8) {
        self.permutation
            .swap(cubicle_a as usize, cubicle_b as usize);
    }

    pub fn parity(&self) -> bool {
        let mut p = true;
        for i in 0..self.permutation.len() {
            for j in (i + 1)..self.permutation.len() {
                if self.permutation[i] > self.permutation[j] {
                    p = !p;
                }
            }
        }
        p
    }

    pub fn cubie_in_cubicle(&self, idx: u8) -> u8 {
        self.permutation[idx as usize]
    }

    /// Get a reference to the permutation's permutation.
    #[must_use]
    pub fn permutation(&self) -> Vec<u8> {
        self.permutation.to_vec()
    }

    /// Set the permutation's permutation.
    pub fn set_permutation(&mut self, permutation: Vec<u8>) -> Result<(), CubeError> {
        if permutation.len() != self.permutation.len() {
            return Err(CubeError::InvalidState);
        }
        self.permutation = permutation;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubies::{NUM_CORNERS, NUM_EDGES};

    impl Permutation {
        pub fn new(size: u8) -> Permutation {
            Permutation {
                permutation: (0..size).collect(),
            }
        }
    }

    #[test]
    fn swap_four_corners_sanity_test() {
        let mut permutation = Permutation::new(NUM_CORNERS);

        permutation.swap_four_cubies(1, 5, 6, 2);
        permutation.swap_four_cubies(1, 2, 6, 5);

        let swapped_permutation =
            Permutation::new_with_permutation(&vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8]);

        assert_eq!(permutation, swapped_permutation);
    }
    #[test]
    fn swap_four_corners_test() {
        let mut permutation = Permutation::new(NUM_CORNERS);

        permutation.swap_four_cubies(1, 5, 6, 2);

        let swapped_permutation =
            Permutation::new_with_permutation(&vec![0u8, 2u8, 6u8, 3u8, 4u8, 1u8, 5u8, 7u8]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_four_corners_twice_test() {
        let mut permutation = Permutation::new(NUM_CORNERS);

        permutation.swap_four_cubies(2, 6, 7, 3); // F turn
        permutation.swap_four_cubies(3, 0, 4, 7); // L' turn

        let swapped_permutation =
            Permutation::new_with_permutation(&vec![7u8, 1u8, 3u8, 6u8, 0u8, 5u8, 2u8, 4u8]);
        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_two_corner_sanity_test() {
        let mut permutation = Permutation::new(NUM_CORNERS);

        permutation.swap_two_cubies(0, 1);
        permutation.swap_two_cubies(0, 1);

        let swapped_permutation =
            Permutation::new_with_permutation(&vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_two_corner_test() {
        let mut permutation = Permutation::new(NUM_CORNERS);

        permutation.swap_two_cubies(0, 1);

        let swapped_permutation =
            Permutation::new_with_permutation(&vec![1u8, 0u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_four_edges_sanity_test() {
        let mut permutation = Permutation::new(NUM_EDGES);

        permutation.swap_four_cubies(1, 5, 6, 2); // R turn
        permutation.swap_four_cubies(1, 2, 6, 5); // R' turn

        let swapped_permutation = Permutation::new_with_permutation(&vec![
            0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8,
        ]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_four_edges_test() {
        let mut permutation = Permutation::new(NUM_EDGES);

        permutation.swap_four_cubies(0, 1, 2, 3); // R turn

        let swapped_permutation = Permutation::new_with_permutation(&vec![
            3u8, 0u8, 1u8, 2u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8,
        ]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_four_edges_twice_test() {
        let mut permutation = Permutation::new(NUM_EDGES);

        permutation.swap_four_cubies(0, 1, 2, 3); // U turn
        permutation.swap_four_cubies(1, 5, 9, 6); // R turn

        let swapped_permutation = Permutation::new_with_permutation(&vec![
            3u8, 6u8, 1u8, 2u8, 4u8, 0u8, 9u8, 7u8, 8u8, 5u8, 10u8, 11u8,
        ]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_two_edges_sanity_test() {
        let mut permutation = Permutation::new(NUM_EDGES);

        permutation.swap_two_cubies(0, 1);
        permutation.swap_two_cubies(0, 1);

        let swapped_permutation = Permutation::new_with_permutation(&vec![
            0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8,
        ]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn swap_two_edges_test() {
        let mut permutation = Permutation::new(NUM_EDGES);

        permutation.swap_two_cubies(0, 1);

        let swapped_permutation = Permutation::new_with_permutation(&vec![
            1u8, 0u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8,
        ]);

        assert_eq!(permutation, swapped_permutation);
    }

    #[test]
    fn parity_test() {
        let mut edge_permutation = Permutation::new(NUM_EDGES);
        let mut corner_permutation = Permutation::new(NUM_CORNERS);

        // U turn
        corner_permutation.swap_four_cubies(0, 1, 2, 3);
        edge_permutation.swap_four_cubies(0, 1, 2, 3);
        // L'
        corner_permutation.swap_four_cubies(3, 0, 4, 7);
        edge_permutation.swap_four_cubies(3, 4, 11, 7);
        // D
        corner_permutation.swap_four_cubies(4, 5, 7, 6);
        edge_permutation.swap_four_cubies(8, 9, 10, 11);

        assert_eq!(corner_permutation.parity(), edge_permutation.parity());
    }
}
