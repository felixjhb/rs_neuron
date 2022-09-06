use std::io::ErrorKind;
use std::thread::current;
use crate::neural_layer::NeuralLayer;
use crate::activation_function::ActivationFn;

type DataPoint = Vec<f32>;
type DataPair<'a> = &'a (DataPoint, DataPoint);
type DataCollection<'a> = &'a Vec<(DataPoint, DataPoint)>;

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

    pub fn evaluate(&self, input: &DataPoint) -> Result<DataPoint, ErrorKind> {
        let mut iteration = input.to_vec();
        for layer in &self.layers {
            iteration = layer.evaluate(iteration)?.into_iter().map(|(x, _)| x).collect();
        }
        Ok(iteration)
    }

    fn evaluate_with_memory(&self, input: &DataPoint) -> Result<Vec<Vec<(f32, f32)>>, ErrorKind> {
        let mut iteration = input.to_vec();
        let mut output = Vec::<Vec<(f32, f32)>>::with_capacity(self.layers().len());
        for layer in &self.layers {
            let current_memory = layer.evaluate(iteration)?;
            iteration = current_memory.to_vec().into_iter().map(|(x, _)| x).collect();
            output.push(current_memory);
        }
        Ok(output)
    }

    fn sqrd_distance(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        x.iter()
        .zip(
            y.iter()
        )
        .map(|(x,y)| (x-y)*(x-y))
        .sum()
    }

    fn cost_at_point(&self, (fx, y): DataPair) -> Result<f32, ErrorKind> {
        assert!(fx.len() == y.len());
        let test_result = self.evaluate(&fx)?;
        if test_result.len() == y.len() {
            Ok(Self::sqrd_distance(&y, &test_result))
        } else {
            Err(ErrorKind::InvalidData)
        }
    }

    pub fn cost (&self, test_data: DataCollection) -> Result<f32, ErrorKind> {
        let data_len = test_data.len();
        let mut running_cost = 0f32;

        for (x, y) in test_data {
            let fx = &self.evaluate(x)?;
            running_cost += self.cost_at_point(&(fx.to_vec(), y.to_vec()))?;
        }
        Ok(running_cost)
    }

    fn update_gradients_every_layer(&mut self, (x, y): DataPair) {
        let entire_history = self.evaluate_with_memory(x).unwrap();

        let final_layer_ref = self.layers_mut().last_mut();
        let final_layer = match final_layer_ref {
            Some(x) => x,
            None => return
        };
        let last_memory = entire_history.last().unwrap();
        let training_vector = final_layer.calculate_final_layer_training_vector(y, &last_memory).unwrap();
        final_layer.update_gradients(&training_vector, last_memory);

        //loop backwards through all layers, ignoring the final layer
        for current_index in (0..self.layers.len() - 1).rev() {
            let layers_mut = self.layers_mut();
            let current_memory = entire_history.get(current_index).unwrap();
            //to avoid memory issues we use split_at_mut's unsafe code to solve this issue by generating new lists of references shared with original list
            //terrible hacky attempt at fix starts
            let (_, hidden_to_final_list) = layers_mut.split_at_mut(current_index);
            let (exactly_hidden_list, prior_to_final_list) = hidden_to_final_list.split_at_mut(1);
            let hidden_layer = exactly_hidden_list.get_mut(0).unwrap();
            let prior_layer = prior_to_final_list.get_mut(0).unwrap();
            //terrible hacky attempt at fix ends
            let new_training_vector = hidden_layer.calculate_hidden_layer_training_vector(&prior_layer, &training_vector, current_memory).unwrap();
            hidden_layer.update_gradients(&new_training_vector, current_memory);
        }
    }

    fn apply_all_gradients(&mut self, training_rate: &f32) {
        for layer in self.layers_mut() {
            layer.apply_gradient(training_rate);
        }
    }

    fn clear_all_gradients(&mut self) {
        for layer in self.layers_mut() {
            layer.clear_gradient();
        }
    }

    pub fn learn(&mut self, training_data: &Vec<(DataPoint, DataPoint)>, training_rate: &f32) {
        for data in training_data {
            self.update_gradients_every_layer(data);
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
        let x = vec![1f32,2f32,3f32];
        let y = vec![3f32,2f32,1f32];
        assert_eq!(NeuralNetwork::sqrd_distance(&x, &y), 8f32)
    }
}