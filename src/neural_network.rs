use std::io::ErrorKind;
pub(crate) mod neural_layer;
use self::neural_layer::{NeuralLayer, activation_function::ActivationFn};

type DataPoint = Vec<f32>;

pub struct NeuralNetwork {
    layers: Vec<NeuralLayer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>, activation_fns: Vec<ActivationFn>) -> Result<NeuralNetwork, ErrorKind> {
        let length = layer_sizes.len();
        if length == 0 {
            Err(ErrorKind::InvalidInput)
        } else {
            let mut layers: Vec<NeuralLayer> = Vec::with_capacity(length);
            for n in 0..length-1 {
                let input_size = layer_sizes.get(n).unwrap();
                let output_size = layer_sizes.get(n+1).unwrap();
                let activation_fn = activation_fns.get(n).unwrap();
                let new_layer = NeuralLayer::new(*input_size, *output_size, *activation_fn);
                layers.push(new_layer);
            }
            Ok(NeuralNetwork {layers})
        }
    }

    fn layers(&self) -> &Vec<NeuralLayer> {
        &self.layers
    }

    fn layers_mut(&mut self) -> &mut Vec<NeuralLayer> {
        &mut self.layers
    }

    pub fn eval(&self, input: DataPoint) -> Result<DataPoint, ErrorKind> {
        let mut iteration = input;
        for layer in &self.layers {
            iteration = layer.evaluate(iteration)?;
        }
        Ok(iteration)

    }

    fn sqrd_distance(x: Vec<f32>, y: Vec<f32>) -> f32 {
        x.iter()
        .zip(
            y.iter()
        )
        .map(|(x,y)| (x-y)*(x-y))
        .sum()
    }

    fn cost_at_point(&self, test_point: DataPoint, actual_result: DataPoint) -> Result<f32, ErrorKind> {
        let test_result = self.eval(test_point)?;
        if test_result.len() == actual_result.len() {
            Ok(Self::sqrd_distance(actual_result, test_result))
        } else {
            Err(ErrorKind::InvalidData)
        }
    }

    pub fn cost (&self, test_data: Vec<DataPoint>, actual_results: Vec<DataPoint>) -> Result<f32, ErrorKind> {
        let data_len = test_data.len();
        if data_len == actual_results.len() {
            let mut running_cost = 0f32;
            for n in 0..data_len {
                let test_datum = test_data.get(n).unwrap();
                let test_result = self.eval(test_datum.to_vec())?;
                let actual_result = actual_results.get(n).unwrap();
                running_cost += Self::cost_at_point(&self, test_result.to_vec(), actual_result.to_vec())?;
            }
            Ok(running_cost)
        } else {
            Err(ErrorKind::InvalidData)
        }
    }

    fn update_all_gradients(&mut self, data: DataPoint, expected_result: DataPoint) {
        self.eval(data);
        let final_layer_ref = self.layers_mut().last_mut();
        let final_layer = match final_layer_ref {
            Some(x) => x,
            None => return
        };
        let training_vector = final_layer.calculate_final_layer_training_vector(expected_result).unwrap();
        final_layer.update_gradients(&training_vector);

        //loop backwards through all layers, ignoring the final layer
        for current_index in (0..self.layers.len() - 1).rev() {
            let hidden_layer = self.layers_mut().get_mut(current_index).unwrap();
            let prior_layer = self.layers().get(current_index + 1).unwrap();
            let new_training_vector = hidden_layer.calculate_hidden_layer_training_vector(&prior_layer, &training_vector).unwrap();
            hidden_layer.update_gradients(&new_training_vector);
        }
    }

    fn apply_all_gradients(&mut self, training_rate: f32) {
        for layer in self.layers_mut() {
            layer.apply_gradient(training_rate);
        }
    }

    fn clear_all_gradients(&mut self) {
        for layer in self.layers_mut() {
            layer.clear_gradient();
        }
    }

    pub fn learn(&mut self, training_data: Vec<(DataPoint, DataPoint)>, training_rate: f32) {
        for (data_point, expected_result) in training_data {
            self.update_all_gradients(data_point, expected_result);
        }

        //Apply all gradients to weights and biases
        self.apply_all_gradients(training_rate);

        //Clear out gradients for next training session
        self.clear_all_gradients();
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn test_sqrd_distance() {
        assert_eq!(NeuralNetwork::sqrd_distance(vec![1f32,2f32,3f32], vec![3f32,2f32,1f32]), 8f32)
    }
}