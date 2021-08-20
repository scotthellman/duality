pub mod tensor;
pub mod layers;
use tensor::DualNumber;
use layers::{Linear, Sigmoid, Layer};

fn ex(x1: DualNumber, x2: DualNumber) -> DualNumber {
    let a = DualNumber::from(3.0);
    let b = DualNumber::from(5.0);

    a*x1*x1 + b*x1*x2
}

fn ex2() -> DualNumber {
    let weights: Vec<f64> = vec![0.5, 0.2, 0.1];
    let weights = weights.into_iter().map(DualNumber::from).collect();
    let linear = Linear{in_size: 3, out_size: 1, weights};
    let sigmoid = Sigmoid{};
    let values: Vec<f64> = vec![1.0, 2.0, 3.0];
    let mut values: Vec<DualNumber> = values.into_iter().map(DualNumber::from).collect();
    values[0].dual = 1.0;

    sigmoid.forward(&linear.forward(&values))[0]

}


fn main() {
    let result = ex2();
    println!("{}", result);
}
