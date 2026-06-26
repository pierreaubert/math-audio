#![doc = include_str!("../README.md")]
#![doc = include_str!("../REFERENCES.md")]
#![allow(unused)]

pub use functions::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub mod functions;

mod get;
mod misc;
mod types;

pub use get::*;
pub use misc::*;
pub use types::*;
