use std::io::ErrorKind;
pub(crate) mod vec_retrieve;
pub(crate) mod activation_function;
use self::vec_retrieve::retrieve;
use self::activation_function::ActivationFn;

pub struct NeuralLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    activation_fn: ActivationFn,
    training_activations: Vec<f32>,
    training_weighted_inputs: Vec<f32>,
    cost_gradient_weight: Vec<Vec<f32>>,
    cost_gradient_bias: Vec<f32>,
}

impl NeuralLayer {
    pub fn new(input_size: usize, output_size: usize, activation_fn: ActivationFn) -> NeuralLayer {
        let weights = vec![vec![0f32; input_size]; output_size];
        let biases = vec![0f32; output_size];
        let training_activations: Vec<f32> = Vec::with_capacity(output_size);
        let training_weighted_inputs: Vec<f32> = Vec::with_capacity(output_size);
        let cost_gradient_weight: Vec<Vec<f32>> = vec![Vec::with_capacity(input_size); output_size];
        let cost_gradient_bias: Vec<f32> = Vec::with_capacity(output_size);
        NeuralLayer {
            input_size,
            output_size,
            weights,
            biases,
            activation_fn,
            training_activations,
            training_weighted_inputs,
            cost_gradient_weight,
            cost_gradient_bias,
        }
    }

    pub fn evaluate(&self, input: Vec<f32>) -> Result<Vec<f32>, ErrorKind> {
        if self.input_size == input.len() {
            let mut output = Vec::with_capacity(self.output_size);

            for neuronOut in 0..self.output_size {
                let mut running_sum = 0f32;
                running_sum += retrieve(&self.biases, neuronOut)?;
                
                for neuronIn in 0..self.input_size {
                    let current_weights = retrieve(&self.weights, neuronOut)?;
                    running_sum += retrieve(&input, neuronIn)? 
                        * retrieve(&current_weights, neuronIn)?;
                }

                output.push(self.activation_fn.evaluate(running_sum));
            }
            Ok(output)
        } else {
            Err(ErrorKind::InvalidInput)
        }
    }

    fn neuron_cost_derivative(activation: &f32, result: &f32) -> f32 {
        2f32 * (activation - result)
    }

    pub fn calculate_hidden_layer_training_vector(&self, prior_layer: &NeuralLayer, prior_training_vector: &Vec<f32>) 
        -> Result<Vec<f32>, ErrorKind> {
        let mut new_training_vector = Vec::with_capacity(self.output_size);
        
        for new_index in 0..new_training_vector.len() {
            let mut new_training_gradient = 0f32;
            for prior_index in 0..prior_training_vector.len() {
                let derivative_weighted_input = prior_layer.weights.get(prior_index).unwrap().get(new_index).unwrap();
                let prior_training_gradient = prior_training_vector.get(prior_index).unwrap();
                new_training_gradient += derivative_weighted_input * prior_training_gradient;
            }

            new_training_gradient *= self.activation_fn.evaluate_derivative(*self.training_weighted_inputs.get(new_index).unwrap());
            new_training_vector.push(new_training_gradient);
        }
        Ok(new_training_vector)
    }

    pub fn calculate_final_layer_training_vector(&self, expected_results: Vec<f32>) -> Result<Vec<f32>, ErrorKind> {
        let length = expected_results.len();
        let mut training_vector = Vec::with_capacity(length);
        
        for i in 0..length {
            let training_activation = retrieve(&self.training_activations, i)?;
            let training_weighted_input = retrieve(&self.training_weighted_inputs, i)?;
            let expected_result = retrieve(&expected_results, i)?;
            let cost_derivative = Self::neuron_cost_derivative(training_activation, expected_result);
            let activation_derivative = self.activation_fn.evaluate_derivative(*training_weighted_input);
            training_vector.push(activation_derivative * cost_derivative);
        }

        Ok(training_vector)
    }

    pub fn update_gradients(&mut self, training_vector: &Vec<f32>) {
        for output_index in 0..self.output_size {
            let training_gradient = retrieve(&training_vector, output_index).unwrap();
            for input_index in 0..self.input_size {
                let training_weighted_input = retrieve(&self.training_weighted_inputs, input_index).unwrap();
                let derivative_cost_wrt_weight = training_weighted_input * training_gradient;
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

    pub fn apply_gradient(&mut self, training_rate: f32) {
        for (weight_vector, (bias, (weight_gradient_vector, bias_gradient))) in 
            self.weights.iter_mut().zip(self.biases.iter_mut().zip(self.cost_gradient_weight.iter_mut().zip(self.cost_gradient_bias.iter_mut())))
        {
            *bias += *bias_gradient * training_rate;
            for (weight, weight_gradient) in weight_vector.iter_mut().zip(weight_gradient_vector.iter_mut()) {
                *weight += *weight_gradient * training_rate;
            }
        }
    }

    pub fn clear_gradient(&self) {

    }
}