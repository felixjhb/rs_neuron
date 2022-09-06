use std::fmt::Display;

use libm::{expf, sinf, cosf};

#[derive(Copy, Clone)]
pub(crate) struct ActivationFn {
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

#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Identity,
    Step,
    InvExp,
    Sine,
    Custom(fn(f32) -> f32, fn(f32) -> f32)
}

impl ActivationFunction {
    pub fn this(&self) -> ActivationFn {
        match self {
            ActivationFunction::Identity => 
                ActivationFn::new(|x| x, |_| 1f32),
            ActivationFunction::Step => 
                ActivationFn::new(|x| if x < 0f32 {0f32} else {1f32}, |_| 0f32),
            ActivationFunction::InvExp => 
                ActivationFn::new(|x| 1f32 / (1f32 + expf(-x)), |x| {let y = expf(-x); y / ((1f32+y)*(1f32+y))}),
            ActivationFunction::Sine =>
                ActivationFn::new(|x| sinf(x), |x| cosf(x)),
            ActivationFunction::Custom(f, df) =>
                ActivationFn::new(*f, *df),
        }
    }
}

impl Default for ActivationFunction {
    fn default() -> Self {
        ActivationFunction::Identity
    }
}

// impl Display for ActivationFunction {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "f(x) = {}", match self {
//             ActivationFunction::Identity => 
//                 "x",
//             ActivationFunction::Step => 
//                 "x < 0 ? 0 : 1",
//             ActivationFunction::InvExp => 
//                 "1/(1+exp(x))",
//             ActivationFunction::Sine =>
//                 "sin(x)",
//             ActivationFunction::Custom(_, _) =>
//                 "Custom",
//             }
//         )
//     }
// }