use super::tensor::DualNumber;

pub trait Layer {
    fn forward(&self, input: &Vec<DualNumber>, gradient_index: Option<usize>) -> Vec<DualNumber>;
    fn num_params(&self) -> usize;
}

pub struct Linear {
    pub in_size: usize,
    pub out_size: usize,
    pub weights: Vec<f64>
}

impl Layer for Linear {
    fn forward(&self, input: &Vec<DualNumber>, gradient_index: Option<usize>) -> Vec<DualNumber> {
        let mut output = Vec::with_capacity(self.out_size);
        for column in 0..self.out_size {
            let offset = column * self.in_size;
            let neuron_out = self.weights[offset..offset+self.in_size].iter()
                .zip(input)
                .enumerate()
                .map(|(i, (&w,&v))| {
                    let dual_value = match gradient_index {
                        None => 0.0,
                        Some(j) => {
                            (i == j) as i64 as f64 // TODO: better way than double casting?
                        }
                    };
                    v * DualNumber{real: w, dual: dual_value}
                })
                .sum();
            output.push(neuron_out);
        }
        output
    }

    fn num_params(&self) -> usize {
        self.weights.len()
    }
}

fn sigmoid(x: DualNumber) -> DualNumber {
    let real = 1.0 / (1.0 + x.real.exp());
    let dual = real * (1.0 - real) * x.dual;
    return DualNumber{real, dual}
}

pub struct Sigmoid {
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Vec<DualNumber>, gradient_index: Option<usize>) -> Vec<DualNumber> {
        input.iter().map(|&v| {
            sigmoid(v)
        })
        .collect()
    }

    fn num_params(&self) -> usize {
        0
    }
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>
}

impl Network {
    fn forward(&self, input: &[f64], layer_index: Option<usize>, weight_index: Option<usize>)
            -> Vec<DualNumber> {
        let input: Vec<DualNumber> = input.iter()
            .map(|&x| DualNumber::from(x))
            .collect();
        self.layers.iter().enumerate().fold(input, |layer_in, (i, layer)| {
            if let Some(l_i) = layer_index {
                if i == l_i {
                    return layer.forward(&layer_in, weight_index)
                }
            }
            layer.forward(&layer_in, None)
        })
    }

    fn compute_gradients(&mut self, input: &[f64]) {

    }
}
