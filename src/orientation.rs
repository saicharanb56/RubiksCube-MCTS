use crate::{
    cubies::{NUM_CORNERS, NUM_CORNER_ORIENTATION, NUM_EDGES, NUM_EDGE_ORIENTATION},
    errors::CubeError,
};

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Orientation {
    orientations: Vec<u8>,
    n_orientation: u8,
}

impl Orientation {
    pub fn edge() -> Orientation {
        Self {
            orientations: vec![0; NUM_EDGES as usize],
            n_orientation: NUM_EDGE_ORIENTATION,
        }
    }

    pub fn corner() -> Orientation {
        Self {
            orientations: vec![0; NUM_CORNERS as usize],
            n_orientation: NUM_CORNER_ORIENTATION,
        }
    }

    pub fn orientation_at_index(&self, idx: u8) -> u8 {
        self.orientations[idx as usize] as u8
    }

    pub fn add_one(&mut self, idx: u8) {
        self.orientations[idx as usize] = match self.orientations[idx as usize] {
            0 => 1,
            1 => {
                if self.n_orientation == 2 {
                    0
                } else {
                    2
                }
            }
            2 => {
                if self.n_orientation == 2 {
                    1
                } else {
                    0
                }
            }
            _ => panic!("invalid orientation encountered. panicing!"),
        }
    }

    pub fn add_two(&mut self, idx: u8) {
        self.orientations[idx as usize] = match self.orientations[idx as usize] {
            0 => 2,
            1 => {
                if self.n_orientation == 2 {
                    1
                } else {
                    0
                }
            }
            2 => {
                if self.n_orientation == 2 {
                    0
                } else {
                    1
                }
            }
            _ => panic!("invalid orientation encountered. panicing!"),
        }
    }

    pub fn sum(&self) -> u8 {
        self.orientations.iter().sum()
    }

    /// Get a reference to the orientation's orientations.
    #[must_use]
    pub fn orientations(&self) -> Vec<u8> {
        self.orientations.to_vec()
    }

    /// Set the orientation's orientations.
    pub fn set_orientations(&mut self, orientations: Vec<u8>) -> Result<(), CubeError> {
        if orientations.len() != self.orientations.len() {
            return Err(CubeError::InvalidState);
        }
        self.orientations = orientations;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Orientation;
    use crate::cubies::{
        Corner, Edge, CORNER_CUBIES, EDGE_CUBIES, NUM_CORNERS, NUM_CORNER_ORIENTATION, NUM_EDGES,
        NUM_EDGE_ORIENTATION,
    };

    impl Orientation {
        pub fn edge_by_index(&self, idx: u8) -> &Edge {
            &EDGE_CUBIES[idx as usize]
        }

        pub fn corner_by_index(&self, idx: u8) -> &Corner {
            &CORNER_CUBIES[idx as usize]
        }

        pub fn new_with_orientation(o: Vec<u8>) -> Orientation {
            assert!(o.len() == NUM_EDGES as usize || o.len() == NUM_CORNERS as usize);
            Orientation {
                n_orientation: if o.len() == NUM_EDGES as usize {
                    NUM_EDGE_ORIENTATION
                } else {
                    NUM_CORNER_ORIENTATION
                },
                orientations: o,
            }
        }
    }

    #[test]
    fn edge_and_orientation_test() {
        let edge_set = Orientation::edge();

        for i in 0..NUM_EDGES {
            let edge = *edge_set.edge_by_index(i);
            let edge_orientation = edge_set.orientation_at_index(i);

            assert_eq!(edge, EDGE_CUBIES[i as usize]);
            assert_eq!(edge_orientation, 0);
        }
    }

    #[test]
    fn first_edge_orientation_test() {
        let edge_set = Orientation::new_with_orientation(vec![1; NUM_EDGES as usize]);

        for i in 0..NUM_EDGES {
            let edge_orientation = edge_set.orientation_at_index(i);
            assert_eq!(edge_orientation, 1);
        }
    }

    #[test]
    fn add_one_edge_test() {
        let mut edge_set = Orientation::new_with_orientation(vec![1; NUM_EDGES as usize]);

        for i in 0..NUM_EDGES {
            let edge_orientation = edge_set.orientation_at_index(i);
            edge_set.add_one(i);
            assert_eq!((edge_orientation + 1) % 2, edge_set.orientation_at_index(i));
        }
    }

    #[test]
    fn edge_sum_test() {
        let edge_set = Orientation::new_with_orientation(vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        assert_eq!(6, edge_set.sum());
    }

    #[test]
    fn corner_and_orientation_test() {
        let corner_set = Orientation::corner();

        for i in 0..NUM_CORNERS {
            let corner = *corner_set.corner_by_index(i);
            let corner_orientation = corner_set.orientation_at_index(i);

            assert_eq!(corner, CORNER_CUBIES[i as usize]);
            assert_eq!(corner_orientation, 0);
        }
    }

    #[test]
    fn first_corner_orientation_test() {
        let corner_set = Orientation::new_with_orientation(vec![2; NUM_CORNERS as usize]);

        for i in 0..NUM_CORNERS {
            let corner_orientation = corner_set.orientation_at_index(i);
            assert_eq!(corner_orientation, 2);
        }
    }

    #[test]
    fn second_orientation_test() {
        let corner_set = Orientation::new_with_orientation(vec![1; NUM_CORNERS as usize]);

        for i in 0..NUM_CORNERS {
            let corner_orientation = corner_set.orientation_at_index(i);
            assert_eq!(corner_orientation, 1);
        }
    }

    #[test]
    fn corner_add_one_test() {
        let mut corner_set = Orientation::new_with_orientation(vec![1; NUM_CORNERS as usize]);

        for i in 0..NUM_CORNERS {
            let corner_orientation = corner_set.orientation_at_index(i);
            corner_set.add_one(i);
            assert_eq!(
                (corner_orientation + 1) % NUM_CORNER_ORIENTATION,
                corner_set.orientation_at_index(i)
            );
        }
    }

    #[test]
    fn add_two_test() {
        let mut corner_set = Orientation::new_with_orientation(vec![1; NUM_CORNERS as usize]);

        for i in 0..NUM_CORNERS {
            let corner_orientation = corner_set.orientation_at_index(i);
            corner_set.add_two(i);
            assert_eq!(
                (corner_orientation + 2) % NUM_CORNER_ORIENTATION,
                corner_set.orientation_at_index(i)
            );
        }
    }

    #[test]
    fn add_test() {
        let mut corner_set = Orientation::corner();

        corner_set.add_one(2);
        corner_set.add_two(2);
        corner_set.add_one(2);

        assert_eq!(corner_set.sum(), 1);
    }

    #[test]
    fn corner_sum_test() {
        let corner_set = Orientation::new_with_orientation(vec![2, 1, 0, 1, 2, 0, 2, 1]);
        assert_eq!(9, corner_set.sum());
    }
}
