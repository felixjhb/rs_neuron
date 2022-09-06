use std::{io::ErrorKind, fmt::Display};
use itertools::izip;

use crate::activation_function::{ActivationFn, ActivationFunction};

pub struct NeuralLayer {
    //The size of the input vector this layer accepts, and size of output layer this layer calculates
    input_size: usize,
    output_size: usize,
    //Actual values used to calculate the output of the neural layer given an input
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    //The activation function is a last step applied to the neural layer's output
    activation_fn: ActivationFn,
    //The cost function's gradient - the vector of the next downhill step, size unmonitored
    //Stored across the whole layer for easier calculation
    cost_gradient_weight: Vec<Vec<f32>>,
    cost_gradient_bias: Vec<f32>,
}

impl NeuralLayer {
    pub fn new(input_size: usize, output_size: usize, activation_fn: ActivationFunction) -> NeuralLayer {
        let weights = vec![vec![0f32; input_size]; output_size];
        let biases = vec![0f32; output_size];
        let cost_gradient_weight = vec![vec![0f32; input_size]; output_size];
        let cost_gradient_bias = vec![0f32; output_size];
        NeuralLayer {
            input_size,
            output_size,
            weights,
            biases,
            activation_fn: activation_fn.this(),
            cost_gradient_weight,
            cost_gradient_bias,
        }
    }

    pub fn evaluate(&self, input: Vec<f32>) -> Result<Vec<(f32, f32)>, ErrorKind> {
        //returns vector of (activated output, unactivated output)
        if self.input_size == input.len() {
            let mut output = Vec::with_capacity(self.output_size);

            for neuron_out in 0..self.output_size {
                let mut running_sum = 0f32;
                running_sum += self.biases.get(neuron_out).unwrap();
                
                for neuron_in in 0..self.input_size {
                    let current_weights = self.weights.get(neuron_out).unwrap();
                    running_sum += input.get(neuron_in).unwrap() * current_weights.get(neuron_in).unwrap();
                }

                output.push((self.activation_fn.evaluate(running_sum), running_sum));
            }
            Ok(output)
        } else {
            Err(ErrorKind::InvalidInput)
        }
    }

    pub fn len(&self) -> usize {
        self.output_size
    }

    fn neuron_cost_derivative(activation: &f32, result: &f32) -> f32 {
        2f32 * (activation - result)
    }

    pub(crate) fn calculate_hidden_layer_training_vector(&self, prior_layer: &NeuralLayer, prior_training_vector: &Vec<f32>, memory: &Vec<(f32, f32)>) 
        -> Result<Vec<f32>, ErrorKind> {
        let mut new_training_vector = Vec::with_capacity(self.output_size);
        
        for new_index in 0..self.output_size {
            let mut new_training_gradient = 0f32;
            for prior_index in 0..prior_training_vector.len() {
                let derivative_weighted_input = prior_layer.weights.get(prior_index).unwrap().get(new_index).unwrap();
                let prior_training_gradient = prior_training_vector.get(prior_index).unwrap();
                new_training_gradient += derivative_weighted_input * prior_training_gradient;
            }
            let memory_weighted_inputs = memory.get(new_index).unwrap().1;
            new_training_gradient *= self.activation_fn.evaluate_derivative(memory_weighted_inputs);
            new_training_vector.push(new_training_gradient);
        }
        Ok(new_training_vector)
    }

    pub(crate) fn calculate_final_layer_training_vector(&self, y: &Vec<f32>, memory: &Vec<(f32, f32)>) -> Result<Vec<f32>, ErrorKind> {
        let length = y.len();
        let mut training_vector = Vec::with_capacity(length);
        
        for i in 0..length {
            let memory_activation = memory.get(i).unwrap().0;
            let memory_weighted_input = memory.get(i).unwrap().1;
            let expected_result = y.get(i).unwrap();
            let cost_derivative = Self::neuron_cost_derivative(&memory_activation, expected_result);
            let activation_derivative = self.activation_fn.evaluate_derivative(memory_weighted_input);
            training_vector.push(activation_derivative * cost_derivative);
        }

        Ok(training_vector)
    }

    pub(crate) fn update_gradients(&mut self, training_vector: &Vec<f32>, memory: &Vec<(f32, f32)>) {
        for output_index in 0..self.output_size {
            let training_gradient = training_vector.get(output_index).unwrap();
            for input_index in 0..self.input_size {
                let memory_weighted_input = memory.get(output_index).unwrap().1;
                let derivative_cost_wrt_weight = memory_weighted_input * training_gradient;
                let variable_to_update = self.cost_gradient_weight
                    .get_mut(output_index)
                    .unwrap()
                    .get_mut(input_index)
                    .unwrap();
                *variable_to_update += derivative_cost_wrt_weight;
            }

            let derivative_cost_wrt_bias = 1f32 * training_gradient;
            *self.cost_gradient_bias.get_mut(output_index).unwrap() += derivative_cost_wrt_bias;
        }
    }

    pub(crate) fn apply_gradient(&mut self, training_rate: &f32) {
        for (weight_vector, (bias, (weight_gradient_vector, bias_gradient))) in 
            self.weights.iter_mut().zip(self.biases.iter_mut().zip(self.cost_gradient_weight.iter_mut().zip(self.cost_gradient_bias.iter_mut())))
        {
            *bias += *bias_gradient * training_rate;
            for (weight, weight_gradient) in weight_vector.iter_mut().zip(weight_gradient_vector.iter_mut()) {
                *weight += *weight_gradient * training_rate;
            }
        }
    }

    pub fn clear_gradient(&mut self) {
        self.cost_gradient_weight = vec![vec![0f32; self.input_size]; self.output_size];
        self.cost_gradient_bias = vec![0f32; self.output_size]
    }
}

impl Display for NeuralLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fns_string = String::new();

        for (ws, b) in izip!(&self.weights, &self.biases).peekable() {
            let w_string = {
                let mut w_str = "[ ".to_string();
                for w in ws {
                    w_str.push_str(&format!("{:.3}, ", w));
                }
                w_str.push_str("]");
                w_str
            };
            fns_string.push_str(&format!("{}x+{:.3}", w_string, b));
        }

        write!(f, "[{fns_string}]")
    }
}