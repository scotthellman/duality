use super::tensor::DualNumber;

pub trait Layer {
    fn forward(&self, input: &Vec<DualNumber>, gradient_index: Option<usize>) -> Vec<DualNumber>;
    fn num_params(&self) -> usize;
    fn update_param(&mut self, increment: f64, index: usize);
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

    fn update_param(&mut self, increment: f64, index: usize) {
        self.weights[index] += increment;
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

    fn update_param(&mut self, increment: f64, index: usize) {
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParamIndex {
    layer: usize,
    param: usize
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    gradients: Option<Vec<f64>>,
    num_params: usize
}

impl Network {
    pub fn from_layers(layers: Vec<Box<dyn Layer>>) -> Network {
        let num_params = layers.iter().map(|l| l.num_params()).sum();
        Network{
            layers,
            num_params,
            gradients: None
        }
    }
    pub fn forward(&self, input: &[f64], param_index: Option<ParamIndex>) -> Vec<DualNumber> {
        let input: Vec<DualNumber> = input.iter()
            .map(|&x| DualNumber::from(x))
            .collect();
        self.layers.iter().enumerate().fold(input, |layer_in, (i, layer)| {
            if let Some(ParamIndex{param: p_i, layer: l_i}) = param_index {
                if i == l_i {
                    return layer.forward(&layer_in, Some(p_i))
                }
            }
            layer.forward(&layer_in, None)
        })
    }

    fn param_iter(&self) -> Vec<ParamIndex> {
        // TODO: this should be a proper iterator
        self.layers.iter().enumerate()
            .map(|(i, l)| {
                (0..l.num_params()).map(move |j| {
                    ParamIndex{param: j, layer: i}
                })
            })
            .flatten()
            .collect()
    }

    fn compute_gradients(&mut self, input: &[f64]) {
        let gradients = self.param_iter().iter()
            .map(|&index| {
                let values: Vec<f64> = self.forward(input, Some(index)).iter()
                    .map(|x| x.dual)
                    .collect();
                values.into_iter()
            })
            .flatten()
            .collect();
        self.gradients = Some(gradients);
    }

    fn update_param(&mut self, increment: f64, index: ParamIndex) {
        self.layers[index.layer].update_param(increment, index.param)
    }

    fn flat_index(&self, index: ParamIndex) -> usize {
        let mut flattened = index.param;
        for (i, l) in self.layers.iter().enumerate() {
            if i < index.layer {
                flattened += l.num_params();
            }
            else {
                break;
            }
        }
        flattened
    }
}

pub fn sgd(network: &mut Network, input: &[f64], target: &[f64], step_size: f64) {
    let result = network.forward(input, None);
    network.compute_gradients(input);
    let loss: Vec<f64> = result.iter().zip(target)
        .map(|(d, t)| {
            (d.real - t) * (d.real - t)
        })
        .collect();
    let param_iter = network.param_iter();
    for index in param_iter {
        let flat = network.flat_index(index);
        let update: f64 = {
            let gradient_chunk = &network.gradients.as_ref().unwrap()[flat..flat + loss.len()];
            gradient_chunk.iter().zip(loss.iter())
                .map(|(g, l)| g*l*step_size)
                .sum()
        };
        network.update_param(update, index);
    }
}
