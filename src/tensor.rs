use std::ops::{Add, Mul, Div, Sub};

#[derive(Clone, Copy, Debug)]
pub struct DualNumber {
    pub real: f64,
    pub dual: f64
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
