use libm::expf;

#[derive(Copy, Clone)]
pub struct ActivationFn {
    funct: fn(f32) -> f32,
    derivative: fn(f32) -> f32,
}

impl ActivationFn {
    pub fn new(funct: fn(f32) -> f32, derivative: fn(f32) -> f32) -> Self {
        Self {funct, derivative}
    }

    pub fn evaluate(&self, x: f32) -> f32 {
        (self.funct)(x)
    }

    pub fn evaluate_derivative(&self, x: f32) -> f32 {
        (self.derivative)(x)
    }
}

pub enum ActivationFunction {
    Identity,
    Step,
    InverseExponential,
}

impl ActivationFunction {
    pub fn this(&self) -> ActivationFn {
        match self {
            ActivationFunction::Identity => 
                ActivationFn::new(|x| x, |_| 1f32),
            ActivationFunction::Step => 
                ActivationFn::new(|x| if x < 0f32 {0f32} else {1f32}, |_| 0f32),
            ActivationFunction::InverseExponential => 
                ActivationFn::new(|x| 1f32 / (1f32 + expf(-x)), |x| {let y = expf(-x); y / ((1f32+y)*(1f32+y))}),
        }
    }
}