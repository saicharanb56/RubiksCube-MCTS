use std::fmt::Display;

use std::str::FromStr;

use rand::distributions::{Distribution, Uniform};

use crate::{
    cubies::*,
    errors::CubeError,
    moves::{MetricKind, Turn},
    orientation::Orientation,
    permutation::Permutation,
};

#[derive(Debug, PartialEq, PartialOrd, Clone)]
// Cube object simulation a 3x3x3 Rubik's Cube
pub struct Cube {
    edge_orientation: Orientation,
    corner_orientation: Orientation,
    edge_permutation: Permutation,
    corner_permutation: Permutation,
    turn_metric: MetricKind,
}

impl Cube {
    pub fn new(turn_metric: MetricKind) -> Cube {
        Cube {
            edge_orientation: Orientation::edge(),
            corner_orientation: Orientation::corner(),
            edge_permutation: Permutation::edge(),
            corner_permutation: Permutation::corner(),
            turn_metric,
        }
    }

    pub fn cube_htm() -> Cube {
        Cube {
            edge_orientation: Orientation::edge(),
            corner_orientation: Orientation::corner(),
            edge_permutation: Permutation::edge(),
            corner_permutation: Permutation::corner(),
            turn_metric: MetricKind::HalfTurnMetric,
        }
    }

    pub fn cube_qtm() -> Cube {
        Cube {
            edge_orientation: Orientation::edge(),
            corner_orientation: Orientation::corner(),
            edge_permutation: Permutation::edge(),
            corner_permutation: Permutation::corner(),
            turn_metric: MetricKind::QuarterTurnMetric,
        }
    }

    /// Initializes a Cube object with values from 6 x 3 x 3 array of facelet colors.
    ///
    /// Initializes a Cube object from the provided array of facelet colors.
    /// The function assumes Green face to the front and Yellow facing up.
    /// The expected order of face colours is W, Y, G, B, R, O
    ///
    /// # Arguments
    ///
    /// * `cube_array` - 6 x 3 x 3 array of face colors
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::rubikscube::{Cube, MetricKind};
    /// let cube_array =
    /// [[["W";3];3],[["Y";3];3],[["G";3];3],[["B";3];3],[["R";3];3],[["O";3];3]];
    /// let cube = Cube::cube_from_array(&cube_array, MetricKind::HalfTurnMetric);
    /// ```
    pub fn cube_from_array(
        cube_array: &[[[&str; 3]; 3]; 6],
        turn_metric: MetricKind,
    ) -> Result<Cube, CubeError> {
        let mut cube_faces = [[[Faces::White; 3]; 3]; 6];

        for (i, face) in cube_array.iter().enumerate() {
            for (j, row) in face.into_iter().enumerate() {
                for (k, elem) in row.iter().enumerate() {
                    let face: Faces = Faces::from_str(elem)?;
                    if j == 1 && k == 1 && face as usize != i {
                        return Err(CubeError::InvalidFaceOrder(face, i as usize));
                    }
                    cube_faces[i][j][k] = face;
                }
            }
        }

        Ok(Cube::cube_from_faces(&cube_faces, turn_metric))
    }

    /// Initializes a Cube object with values from 6 x 3 x 3 array of Face instances.
    ///
    /// Helper function for cube_from_array.
    ///
    /// # Arguments
    ///
    /// * `cube_faces` - 6 x 3 x 3 array of Face instances
    ///
    fn cube_from_faces(cube_faces: &[[[Faces; 3]; 3]; 6], turn_metric: MetricKind) -> Cube {
        let mut edge_permutation = Vec::new();
        let mut corner_permutation = Vec::new();
        let mut edge_orientation = Orientation::edge();
        let mut corner_orientation = Orientation::corner();

        // sets corner cubies orientation and values
        for corner_idx in CORNER_FACELETS.iter() {
            let (primary_facelet_idx, secondary_facelet_idx, tertiary_facelet_idx) = corner_idx;

            let facelet_a = cube_faces[primary_facelet_idx.0 as usize]
                [primary_facelet_idx.1 as usize][primary_facelet_idx.2 as usize];
            let facelet_b = cube_faces[secondary_facelet_idx.0 as usize]
                [secondary_facelet_idx.1 as usize][secondary_facelet_idx.2 as usize];
            let facelet_c = cube_faces[tertiary_facelet_idx.0 as usize]
                [tertiary_facelet_idx.1 as usize][tertiary_facelet_idx.2 as usize];

            let corner_cubie = Corner::new(facelet_a, facelet_b, facelet_c);

            let corner_cubie_idx = corner_cubie.cubie_index();
            corner_permutation.push(corner_cubie_idx);

            let primary_facelet: Faces = CORNER_CUBIES[corner_cubie_idx as usize].facelet_a();

            match corner_cubie.get_orientation(primary_facelet) {
                1 => corner_orientation.add_two(corner_cubie_idx),
                2 => corner_orientation.add_one(corner_cubie_idx),
                _ => {}
            }
        }

        // sets edge cubies orientation and values
        for edge_idx in EDGE_FACELETS.iter() {
            let (primary_facelet_idx, secondary_facelet_idx) = edge_idx;

            let facelet_a = cube_faces[primary_facelet_idx.0 as usize]
                [primary_facelet_idx.1 as usize][primary_facelet_idx.2 as usize];
            let facelet_b = cube_faces[secondary_facelet_idx.0 as usize]
                [secondary_facelet_idx.1 as usize][secondary_facelet_idx.2 as usize];

            let edge_cubie = Edge::new(facelet_a, facelet_b);

            let edge_cubie_idx = edge_cubie.cubie_index();
            edge_permutation.push(edge_cubie_idx);

            let primary_facelet: Faces = EDGE_CUBIES[edge_cubie_idx as usize].facelet_a();

            match edge_cubie.get_orientation(primary_facelet) {
                1 => edge_orientation.add_one(edge_cubie_idx),
                _ => {}
            }
        }

        let edge_permutation = Permutation::new_with_permutation(&edge_permutation);
        let corner_permutation = Permutation::new_with_permutation(&corner_permutation);

        Cube {
            edge_orientation,
            corner_orientation,
            edge_permutation,
            corner_permutation,
            turn_metric,
        }
    }

    /// creates a scrambled rubiks cube
    ///
    /// # Arguments
    ///
    /// * `num_turns` - numner of turns to scramble the cube
    ///
    /// # Examples
    ///
    /// ```
    /// use rubikscube::Cube;
    /// use rubikscube::MetricKind;
    /// let num_scramble_turns = 100;
    ///
    /// let mut cube = Cube::cube_htm();
    /// cube.scramble(num_scramble_turns);
    /// ```
    pub fn scramble(&mut self, num_turns: u32) {
        let between = Uniform::from(0..self.turn_metric as u8);
        let mut rng = rand::thread_rng();

        for _ in 0..num_turns {
            let sampled_index: u8 = between.sample(&mut rng);
            let sampled_turn: Turn = Turn::from_u8(sampled_index).unwrap();
            self._turn(sampled_turn);
        }
    }
    /// Performs the specified turn on the cube object.
    ///
    /// # Arguments
    ///
    /// * `twist` - index of turn enum variant
    ///
    /// # ExamplesCube
    ///
    /// ```
    /// use rubikscube::{Cube, Turn, MetricKind};
    ///
    /// let mut cube = Cube::cube_htm();
    /// cube.turn(0); // Turn L
    /// ```
    pub fn turn(&mut self, twist: u8) -> Result<(), CubeError> {
        if twist >= self.turn_metric as u8 {
            Err(CubeError::InvalidTurn(twist, self.turn_metric as u8))
        } else {
            let t: Turn = Turn::from_u8(twist)?;
            self._turn(t);
            Ok(())
        }
    }

    /// Performs the specified turn on the cube object.
    ///
    /// # Arguments
    ///
    /// * `m` - instance of Turn enum
    ///
    fn _turn(&mut self, m: Turn) {
        // unpack cubicle indices
        let ((a, b, c, d), (w, x, y, z)) = match m {
            Turn::L | Turn::L_ | Turn::L2 => (L_EDGE_CUBICLES, L_CORNER_CUBICLES),
            Turn::R | Turn::R_ | Turn::R2 => (R_EDGE_CUBICLES, R_CORNER_CUBICLES),
            Turn::F | Turn::F_ | Turn::F2 => (F_EDGE_CUBICLES, F_CORNER_CUBICLES),
            Turn::B | Turn::B_ | Turn::B2 => (B_EDGE_CUBICLES, B_CORNER_CUBICLES),
            Turn::U | Turn::U_ | Turn::U2 => (U_EDGE_CUBICLES, U_CORNER_CUBICLES),
            Turn::D | Turn::D_ | Turn::D2 => (D_EDGE_CUBICLES, D_CORNER_CUBICLES),
        };

        // updating edge and corner cubie orientation based on move
        match m {
            Turn::L | Turn::R | Turn::L_ | Turn::R_ => {
                let cubie_a = self.edge_permutation.cubie_in_cubicle(a);
                let cubie_b = self.edge_permutation.cubie_in_cubicle(b);
                let cubie_c = self.edge_permutation.cubie_in_cubicle(c);
                let cubie_d = self.edge_permutation.cubie_in_cubicle(d);
                let cubie_w = self.corner_permutation.cubie_in_cubicle(w);
                let cubie_x = self.corner_permutation.cubie_in_cubicle(x);
                let cubie_y = self.corner_permutation.cubie_in_cubicle(y);
                let cubie_z = self.corner_permutation.cubie_in_cubicle(z);

                self.corner_orientation.add_one(cubie_w);
                self.corner_orientation.add_two(cubie_x);
                self.corner_orientation.add_one(cubie_y);
                self.corner_orientation.add_two(cubie_z);

                self.edge_orientation.add_one(cubie_a);
                self.edge_orientation.add_one(cubie_b);
                self.edge_orientation.add_one(cubie_c);
                self.edge_orientation.add_one(cubie_d);
            }
            Turn::F | Turn::B | Turn::F_ | Turn::B_ => {
                let cubie_w = self.corner_permutation.cubie_in_cubicle(w);
                let cubie_x = self.corner_permutation.cubie_in_cubicle(x);
                let cubie_y = self.corner_permutation.cubie_in_cubicle(y);
                let cubie_z = self.corner_permutation.cubie_in_cubicle(z);

                self.corner_orientation.add_one(cubie_w);
                self.corner_orientation.add_two(cubie_x);
                self.corner_orientation.add_one(cubie_y);
                self.corner_orientation.add_two(cubie_z);
            }
            _ => {}
        }

        // updating edge and corner cubie permutation based on move
        match m {
            Turn::L | Turn::R | Turn::F | Turn::B | Turn::U | Turn::D => {
                self.edge_permutation.swap_four_cubies(a, b, c, d);
                self.corner_permutation.swap_four_cubies(w, x, y, z);
            }
            Turn::L_ | Turn::R_ | Turn::F_ | Turn::B_ | Turn::U_ | Turn::D_ => {
                self.edge_permutation.swap_four_cubies(d, c, b, a);
                self.corner_permutation.swap_four_cubies(z, y, x, w);
            }
            Turn::L2 | Turn::R2 | Turn::F2 | Turn::B2 | Turn::U2 | Turn::D2 => {
                self.edge_permutation.swap_two_cubies(a, c);
                self.edge_permutation.swap_two_cubies(b, d);
                self.corner_permutation.swap_two_cubies(w, y);
                self.corner_permutation.swap_two_cubies(x, z);
            }
        }
    }

    /// Checks if current configuration of cube is solvable.
    /// used to check cube objects created with cube_from_array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rubikscube::Cube;
    /// use rubikscube::MetricKind;
    ///
    ///    let cube_array = [
    ///        [["O", "Y", "O"], ["G", "W", "B"], ["O", "G", "B"]], // W
    ///        [["B", "W", "R"], ["B", "Y", "Y"], ["R", "O", "G"]], // Y
    ///        [["G", "Y", "Y"], ["B", "G", "R"], ["Y", "R", "B"]], // G
    ///        [["B", "O", "W"], ["G", "B", "R"], ["Y", "O", "W"]], // B
    ///        [["R", "W", "W"], ["G", "R", "R"], ["G", "Y", "G"]], // R
    ///        [["R", "B", "Y"], ["W", "O", "W"], ["W", "O", "O"]], // O
    ///    ];
    ///
    /// let cube = Cube::cube_from_array(&cube_array, MetricKind::HalfTurnMetric).unwrap();
    ///
    /// assert!(cube.is_solvable());
    /// ```
    pub fn is_solvable(&self) -> bool {
        (self.edge_permutation.parity() == self.corner_permutation.parity())
            && (self.edge_orientation.sum() % 2 == 0)
            && (self.corner_orientation.sum() % 3 == 0)
    }

    /// calculates the representation of the cube as a one hot array of size 480.
    ///
    /// calculates the repesentation of the rubik's cube as a one hot array of size 480.
    /// Each edge cubie can be in 12 positions with 2 orientations, therefore 24 possible states.
    /// Each corner cubie can be in 8 positions with 3 orientations, therefore 24 possible states.
    ///
    /// The 20 cubies each with 24 possible states gives us the 480 length one-hot array.
    ///
    pub fn representation(&self) -> [bool; 480] {
        let mut repr = [false; 480];

        // corner cubicles representation
        for corner_idx in 0..NUM_CORNERS {
            let cubie_idx = self.corner_permutation.cubie_in_cubicle(corner_idx);
            let cubie_orientation =
                self.corner_orientation.orientation_at_index(cubie_idx) as usize;
            let index = (NUM_STATES * cubie_idx) as usize
                + (NUM_CORNER_ORIENTATION * corner_idx) as usize
                + cubie_orientation;
            repr[index] = true;
        }

        // edge cubicles representation
        for edge_idx in 0..NUM_EDGES {
            let cubie_idx = self.edge_permutation.cubie_in_cubicle(edge_idx);
            let cubie_orientation = self.edge_orientation.orientation_at_index(cubie_idx) as usize;
            let index = (NUM_STATES as usize * (cubie_idx + NUM_CORNERS) as usize
                + (NUM_EDGE_ORIENTATION * edge_idx) as usize
                + cubie_orientation) as usize;
            repr[index] = true;
        }

        repr
    }

    /// returns the turn metric of the cube
    ///
    /// # Examples
    ///
    /// ```
    /// use rubikscube::Cube;
    /// use rubikscube::MetricKind;
    ///
    ///
    /// let cube = Cube::cube_htm();
    /// assert!(cube.turn_metric() == MetricKind::HalfTurnMetric);
    ///
    /// let another_cube = Cube::cube_qtm();
    /// assert!(another_cube.turn_metric() == MetricKind::QuarterTurnMetric);
    /// ```
    pub fn turn_metric(&self) -> MetricKind {
        self.turn_metric
    }

    /// returns true if cube is solved
    ///
    /// # Examples
    ///
    /// ```
    /// use rubikscube::Cube;
    /// use rubikscube::Turn;
    /// use rubikscube::MetricKind;
    ///
    ///
    /// let mut cube = Cube::cube_htm();
    /// cube.turn(Turn::L as u8);
    /// cube.turn(Turn::F as u8);
    /// cube.turn(Turn::R as u8);
    /// cube.turn(Turn::R_ as u8);
    /// cube.turn(Turn::F_ as u8);
    /// cube.turn(Turn::L_ as u8);
    /// assert!(cube.solved());
    ///
    /// ```
    pub fn solved(&self) -> bool {
        let repr = self.representation();
        let mut solved = true;

        for i in 0..12 {
            // corner check
            if i < 8 {
                solved &= repr[i * 27];
            }
            // edges check
            solved &= repr[i * 26 + 192];
        }

        solved
    }

    pub fn get_state(&self) -> [Vec<u8>; 4] {
        let edge_orientation_state: Vec<u8> = self.edge_orientation.orientations();
        let corner_orientation_state: Vec<u8> = self.corner_orientation.orientations();
        let edge_permutation_state = self.edge_permutation.permutation();
        let corner_permuation_state = self.corner_permutation.permutation();

        [
            edge_orientation_state,
            corner_orientation_state,
            edge_permutation_state,
            corner_permuation_state,
        ]
    }

    pub fn set_state(
        &mut self,
        edge_orienatation_state: Vec<u8>,
        corner_orientation_state: Vec<u8>,
        edge_permutation_state: Vec<u8>,
        corner_permuation_state: Vec<u8>,
    ) -> Result<(), CubeError> {
        self.edge_orientation
            .set_orientations(edge_orienatation_state)?;
        self.corner_orientation
            .set_orientations(corner_orientation_state)?;
        self.edge_permutation
            .set_permutation(edge_permutation_state)?;
        self.corner_permutation
            .set_permutation(corner_permuation_state)?;

        Ok(())
    }
}

impl Display for Cube {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let corner_cubies: Vec<Corner> = (0..NUM_CORNERS)
            .into_iter()
            .map(|idx| {
                let cubie_idx = self.corner_permutation.cubie_in_cubicle(idx);
                (cubie_idx, CORNER_CUBIES[cubie_idx as usize].clone())
            })
            .map(|(idx, corner)| -> Corner {
                corner.orient_corner(self.corner_orientation.orientation_at_index(idx))
            })
            .collect();

        let edge_cubies: Vec<Edge> = (0..NUM_EDGES)
            .into_iter()
            .map(|idx| {
                let cubie_idx = self.edge_permutation.cubie_in_cubicle(idx);
                (cubie_idx, EDGE_CUBIES[cubie_idx as usize].clone())
            })
            .map(|(idx, edge)| -> Edge {
                edge.orient_edge(self.edge_orientation.orientation_at_index(idx))
            })
            .collect();

        let top_face = format!(
            "
                   {} {} {}
                   {} Y {}
                   {} {} {}
                   ",
            corner_cubies[0].facelet_a(),
            edge_cubies[0].facelet_a(),
            corner_cubies[1].facelet_a(),
            edge_cubies[3].facelet_a(),
            edge_cubies[1].facelet_a(),
            corner_cubies[3].facelet_a(),
            edge_cubies[2].facelet_a(),
            corner_cubies[2].facelet_a()
        );

        let middle_faces = format!(
            "
            {} {} {}  {} {} {}  {} {} {}  {} {} {}
            {} R {}  {} G {}  {} O {}  {} B {}
            {} {} {}  {} {} {}  {} {} {}  {} {} {}
                                 ",
            corner_cubies[0].facelet_b(),
            edge_cubies[3].facelet_b(),
            corner_cubies[3].facelet_c(),
            corner_cubies[3].facelet_b(),
            edge_cubies[2].facelet_b(),
            corner_cubies[2].facelet_c(),
            corner_cubies[2].facelet_b(),
            edge_cubies[1].facelet_b(),
            corner_cubies[1].facelet_c(),
            corner_cubies[1].facelet_b(),
            edge_cubies[0].facelet_b(),
            corner_cubies[0].facelet_c(),
            edge_cubies[4].facelet_a(),
            edge_cubies[7].facelet_a(),
            edge_cubies[7].facelet_b(),
            edge_cubies[6].facelet_b(),
            edge_cubies[6].facelet_a(),
            edge_cubies[5].facelet_a(),
            edge_cubies[5].facelet_b(),
            edge_cubies[4].facelet_b(),
            corner_cubies[4].facelet_c(),
            edge_cubies[11].facelet_b(),
            corner_cubies[7].facelet_b(),
            corner_cubies[7].facelet_c(),
            edge_cubies[10].facelet_b(),
            corner_cubies[6].facelet_b(),
            corner_cubies[6].facelet_c(),
            edge_cubies[9].facelet_b(),
            corner_cubies[5].facelet_b(),
            corner_cubies[5].facelet_c(),
            edge_cubies[8].facelet_b(),
            corner_cubies[4].facelet_b(),
        );

        let bottom_face = format!(
            "
                   {} {} {}
                   {} W {}
                   {} {} {}
                   ",
            corner_cubies[7].facelet_a(),
            edge_cubies[10].facelet_a(),
            corner_cubies[6].facelet_a(),
            edge_cubies[11].facelet_a(),
            edge_cubies[9].facelet_a(),
            corner_cubies[4].facelet_a(),
            edge_cubies[8].facelet_a(),
            corner_cubies[5].facelet_a()
        );

        write!(f, "{}{}{}", top_face, middle_faces, bottom_face)
    }
}

#[cfg(test)]
mod tests {

    use crate::cubies::Faces;
    use crate::errors::CubeError;
    use crate::{Cube, MetricKind, Turn};

    #[test]
    fn cube_sanity_test() {
        let cube = Cube::cube_htm();
        assert_eq!(
            cube.edge_permutation.parity(),
            cube.corner_permutation.parity()
        );

        assert_eq!(cube.edge_orientation.sum() % 2, 0);
        assert_eq!(cube.corner_orientation.sum() % 3, 0);
    }
    #[test]
    fn cube_undo_turn_test() {
        let solved_cube = Cube::cube_htm();
        let mut cube = Cube::cube_htm();

        cube._turn(Turn::L);
        cube._turn(Turn::L_);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::L2);
        cube._turn(Turn::L2);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::R);
        cube._turn(Turn::R_);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::R2);
        cube._turn(Turn::R2);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::U);
        cube._turn(Turn::U_);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::U2);
        cube._turn(Turn::U2);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::D);
        cube._turn(Turn::D_);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::D2);
        cube._turn(Turn::D2);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::F);
        cube._turn(Turn::F_);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::F2);
        cube._turn(Turn::F2);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::B);
        cube._turn(Turn::B_);

        assert_eq!(cube, solved_cube);

        cube._turn(Turn::B2);
        cube._turn(Turn::B2);

        assert_eq!(cube, solved_cube);
    }

    #[test]
    fn cube_turns_test() {
        let mut cube = Cube::cube_htm();
        cube.scramble(1000);
        assert!(cube.is_solvable());
    }

    #[test]
    fn smove_test() {
        let mut cube = Cube::cube_htm();
        let solved_cube = Cube::cube_htm();

        for _ in 0..7 {
            cube._turn(Turn::U);
            cube._turn(Turn::R);
            cube._turn(Turn::R_);
            cube._turn(Turn::U_);
        }

        assert_eq!(cube, solved_cube);
    }

    #[test]
    fn cube_solvable_test() {
        let mut cube = Cube::cube_htm();
        cube.scramble(1000);

        assert_eq!(
            cube.edge_permutation.parity(),
            cube.corner_permutation.parity()
        );

        assert_eq!(cube.edge_orientation.sum() % 2, 0);
        assert_eq!(cube.corner_orientation.sum() % 3, 0);
    }

    #[test]
    fn random_scramble() {
        // B' R U2 R U R' L' U2 F B' D2 F2 D' L B2 D2 R2 L D' L D2 R L2 B' R'
        let mut cube = Cube::cube_htm();

        cube._turn(Turn::B_);
        cube._turn(Turn::R);
        cube._turn(Turn::U2);
        cube._turn(Turn::R);
        cube._turn(Turn::U);
        cube._turn(Turn::R_);
        cube._turn(Turn::L_);
        cube._turn(Turn::U2);
        cube._turn(Turn::F);
        cube._turn(Turn::B_);
        cube._turn(Turn::D2);
        cube._turn(Turn::F2);
        cube._turn(Turn::D_);
        cube._turn(Turn::L);
        cube._turn(Turn::B2);
        cube._turn(Turn::D2);
        cube._turn(Turn::R2);
        cube._turn(Turn::L);
        cube._turn(Turn::D_);
        cube._turn(Turn::L);
        cube._turn(Turn::D2);
        cube._turn(Turn::R);
        cube._turn(Turn::L2);
        cube._turn(Turn::B_);
        cube._turn(Turn::R_);

        println!("{}", cube);

        let cube_array = [
            [["O", "Y", "O"], ["G", "W", "B"], ["O", "G", "B"]], // W
            [["B", "W", "R"], ["B", "Y", "Y"], ["R", "O", "G"]], // Y
            [["G", "Y", "Y"], ["B", "G", "R"], ["Y", "R", "B"]], // G
            [["B", "O", "W"], ["G", "B", "R"], ["Y", "O", "W"]], // B
            [["R", "W", "W"], ["G", "R", "R"], ["G", "Y", "G"]], // R
            [["R", "B", "Y"], ["W", "O", "W"], ["W", "O", "O"]], // O
        ];

        let cube_from_array =
            Cube::cube_from_array(&cube_array, MetricKind::HalfTurnMetric).unwrap();

        assert!(cube_from_array.is_solvable());

        println!("{}", cube_from_array);
        assert_eq!(cube_from_array, cube);
    }

    #[test]
    fn from_array_sanity_test() {
        let cube_array = [
            [["W"; 3]; 3],
            [["Y"; 3]; 3],
            [["G"; 3]; 3],
            [["B"; 3]; 3],
            [["R"; 3]; 3],
            [["O"; 3]; 3],
        ];
        let cube = Cube::cube_from_array(&cube_array, MetricKind::HalfTurnMetric).unwrap();

        let solved_cube = Cube::cube_htm();
        assert!(cube.is_solvable());
        assert_eq!(cube, solved_cube);
    }

    #[test]
    fn from_array_order_err_test() {
        let cube_array = [
            [["W"; 3]; 3],
            [["Y"; 3]; 3],
            [["B"; 3]; 3],
            [["G"; 3]; 3],
            [["R"; 3]; 3],
            [["O"; 3]; 3],
        ];
        let cube_err = Cube::cube_from_array(&cube_array, MetricKind::HalfTurnMetric).unwrap_err();

        assert_eq!(cube_err, CubeError::InvalidFaceOrder(Faces::Blue, 2));
    }

    #[test]
    fn from_array_color_err_test() {
        let cube_array = [
            [["P"; 3]; 3],
            [["Y"; 3]; 3],
            [["G"; 3]; 3],
            [["B"; 3]; 3],
            [["R"; 3]; 3],
            [["O"; 3]; 3],
        ];
        let cube_err = Cube::cube_from_array(&cube_array, MetricKind::HalfTurnMetric).unwrap_err();

        assert_eq!(cube_err, CubeError::InvalidFaceletColor);
    }

    #[test]
    fn representation_test() {
        let cube = Cube::cube_htm();

        let expeceted_repr = [
            [
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, true, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, true, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, true, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, true, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, false, false,
            ],
            [
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, true, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, true, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, true, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, true, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, true, false, false, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, false, false, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, true, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, true, false,
            ],
        ];

        let mut expeceted_repr_flattened = [false; 480];

        for (i, row) in expeceted_repr.iter().enumerate() {
            for (j, elem) in row.iter().enumerate() {
                expeceted_repr_flattened[i * 24usize + j] = *elem
            }
        }

        let repr = cube.representation();

        for (x, y) in repr.iter().zip(expeceted_repr_flattened.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn cube_quarter_turn_test() {
        let mut cube = Cube::cube_qtm();
        cube.scramble(1000);
        assert!(cube.is_solvable());
    }
}
