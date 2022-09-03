mod neural_network;
use neural_network::NeuralNetwork;
use neural_network::neural_layer::activation_function::ActivationFunction;


fn main() {

    let my_neural_network = NeuralNetwork::new(vec![2,3,2], vec![ActivationFunction::Identity.this(); 3]).unwrap();
    
    let x = vec![vec![1f32, 0f32], vec![2f32, 0f32], vec![0f32, 3f32]];
    let y = vec![vec![1f32, 1f32], vec![2f32, 1f32], vec![0f32, 1f32]];
    let cost = my_neural_network.cost(x, y).unwrap();
    println!("Total Cost: {}", cost)
}