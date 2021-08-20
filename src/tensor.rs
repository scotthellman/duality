use std::ops::{Add, Mul, Div, Sub};
use std::iter::Sum;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct DualNumber {
    pub real: f64,
    pub dual: f64
}

impl fmt::Display for DualNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}+{}ε", self.real, self.dual)
    }
}


impl Add for DualNumber {
    type Output = DualNumber;

    fn add(self, rhs: DualNumber) -> DualNumber {
        DualNumber{
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual
        }
    }
}

impl Sub for DualNumber {
    type Output = DualNumber;

    fn sub(self, rhs: DualNumber) -> DualNumber {
        DualNumber{
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual
        }
    }
}

impl Div for DualNumber {
    type Output = DualNumber;

    fn div(self, rhs: DualNumber) -> DualNumber {
        let numerator = self * rhs;
        let denominator = rhs.real * rhs.real;
        DualNumber{
            real: numerator.real / denominator,
            dual: numerator.dual / denominator
        }
    }
}


impl Mul for DualNumber {
    type Output = DualNumber;

    fn mul(self, rhs: DualNumber) -> DualNumber {
        DualNumber{
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real
        }
    }
}

impl From<f64> for DualNumber {
    fn from(x: f64) -> DualNumber {
        DualNumber{
            real: x,
            dual: 0.0
        }
    }
}

impl Sum for DualNumber {
    fn sum<I>(iter: I) -> DualNumber 
        where I: Iterator<Item = Self>,
    {
        iter.fold(DualNumber{real: 0.0, dual: 0.0}, |a, b| a+b)
    }
}
