pub mod tensor;
pub mod layers;
use tensor::DualNumber;
use layers::{Linear, Sigmoid, Layer, Network, sgd};

fn ex(x1: DualNumber, x2: DualNumber) -> DualNumber {
    let a = DualNumber::from(3.0);
    let b = DualNumber::from(5.0);

    a*x1*x1 + b*x1*x2
}

fn ex2() {
    let weights: Vec<f64> = vec![0.5, 0.2];
    let linear = Linear{in_size: 2, out_size: 1, weights};
    let sigmoid = Sigmoid{};

    let layers: Vec<Box<dyn Layer>> = vec![Box::new(linear), Box::new(sigmoid)];

    let mut network = Network::from_layers(layers);

    let input = vec![0.0, 1.0];
    let expected = vec![0.3];

    for i in 0..300 {
        sgd(&mut network, &input, &expected, 10.0);
        println!("{}", network.forward(&input, None)[0]);
    }
}


fn main() {
    let result = ex2();
}
