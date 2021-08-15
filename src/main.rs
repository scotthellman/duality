pub mod tensor;
use tensor::DualNumber;

fn ex(x1: DualNumber, x2: DualNumber) -> (DualNumber) {
    let a = DualNumber::from(3.0);
    let b = DualNumber::from(5.0);

    a*x1*x1 + b*x1*x2
}

fn main() {
    let x1 = DualNumber{real: 2., dual: 1.};
    let x2 = DualNumber{real: -1., dual: 0.};
    let first_eval = ex(x1, x2);
    println!("first eval {:?}", first_eval);

    let x1 = DualNumber{real: 2., dual: 0.};
    let x2 = DualNumber{real: -1., dual: 1.};
    let second_eval = ex(x1, x2);
    println!("second eval {:?}", second_eval);

    println!("derivative wrt x1: {}", first_eval.dual);
    println!("derivative wrt x2: {}", second_eval.dual);
}
