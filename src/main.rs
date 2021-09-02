pub mod tensor;
pub mod layers;
use rand::prelude::*;
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

    for i in 0..300 {
        let inputs: Vec<f64> = (0..2).map({|_| 
            if rand::random::<bool>() {
                1.0
            } else{
                0.0
            }
        }).collect();

        let predicate = inputs.iter()
            .any(|&x| {
                x == 1.0
            });
        let expected = if predicate {vec![1.0]} else {vec![0.0]};
        sgd(&mut network, &inputs, &expected, 10.0);
        println!("{:?} -> {}", inputs, network.forward(&inputs, None)[0]);
    }
}


fn main() {
    let result = ex2();
}
