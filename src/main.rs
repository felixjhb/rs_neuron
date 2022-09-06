mod neural_network;
mod neural_layer;
mod activation_function;

use neural_network::NeuralNetwork;
use activation_function::ActivationFunction;


fn main() {

    let mut my_neural_network = NeuralNetwork::new(vec![2,3,2], vec![ActivationFunction::Identity.this(); 3]).unwrap();
    
    let data = vec![
        (vec![1f32, 0f32], vec![1f32, 1f32]),
        (vec![2f32, 0f32], vec![2f32, 1f32]),
        (vec![0f32, 3f32], vec![0f32, 1f32]),
    ];

    let cost = my_neural_network.cost(&data).unwrap();
    println!("Before Training Cost: {}", cost);
    
    my_neural_network.learn(&data, &0.25);

    let cost = my_neural_network.cost(&data).unwrap();
    println!("After Training Cost: {}", cost);
}