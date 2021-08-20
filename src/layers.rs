use super::tensor::DualNumber;

pub trait Layer {
    fn forward(&self, input: &Vec<DualNumber>) -> Vec<DualNumber>;
}

pub struct Linear {
    pub in_size: usize,
    pub out_size: usize,
    pub weights: Vec<DualNumber>
}

impl Layer for Linear {
    fn forward(&self, input: &Vec<DualNumber>) -> Vec<DualNumber> {
        let mut output = Vec::with_capacity(self.out_size);
        for column in 0..self.out_size {
            let offset = column * self.in_size;
            let neuron_out = self.weights[offset..offset+self.in_size].iter()
                .zip(input)
                .map(|(&w,&v)| {
                    w*v
                })
                .sum();
            output.push(neuron_out);
        }
        output
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
    fn forward(&self, input: &Vec<DualNumber>) -> Vec<DualNumber> {
        input.iter().map(|&v| {
            sigmoid(v)
        })
        .collect()
    }
}
