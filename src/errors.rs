use std::fmt;

use crate::cubies::Faces;

#[derive(Debug, PartialEq)]
pub enum CubeError {
    InvalidFaceOrder(Faces, usize),
    InvalidFaceletColor,
    InvalidTurn(u8, u8),
    InvalidState,
}

impl std::error::Error for CubeError {}

impl fmt::Display for CubeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CubeError::InvalidFaceOrder(face_found, index) => write!(
                f,
                "Invalid order found, Expected {} at face index {}, found {}",
                index,
                Faces::from_repr(*index).unwrap(),
                face_found
            ),
            CubeError::InvalidFaceletColor => write!(f, "Invalid Facelet color"),
            CubeError::InvalidTurn(index, limit) => {
                write!(
                    f,
                    "Invalid Turn, Expected int between 0 and {} got {}",
                    limit - 1,
                    index
                )
            }
            CubeError::InvalidState => {
                write!(f, "Invalid State",)
            }
        }
    }
}

impl From<strum::ParseError> for CubeError {
    fn from(_: strum::ParseError) -> Self {
        CubeError::InvalidFaceletColor
    }
}
