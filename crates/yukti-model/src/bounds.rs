use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Sense {
    #[default]
    Minimize,
    Maximize,
}

impl std::fmt::Display for Sense {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sense::Minimize => write!(f, "MINIMIZE"),
            Sense::Maximize => write!(f, "MAXIMIZE"),
        }
    }
}

/// A closed, semi-infinite, or unbounded interval [lower, upper].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bound {
    pub lower: f64,
    pub upper: f64,
}

impl Bound {
    pub const fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    pub const fn free() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    pub const fn non_negative() -> Self {
        Self {
            lower: 0.0,
            upper: f64::INFINITY,
        }
    }

    pub const fn fixed(val: f64) -> Self {
        Self {
            lower: val,
            upper: val,
        }
    }

    pub const fn upper_only(upper: f64) -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper,
        }
    }

    pub const fn lower_only(lower: f64) -> Self {
        Self {
            lower,
            upper: f64::INFINITY,
        }
    }

    pub fn is_fixed(&self, tol: f64) -> bool {
        (self.upper - self.lower).abs() <= tol
    }

    pub fn is_free(&self) -> bool {
        self.lower.is_infinite()
            && self.lower.is_sign_negative()
            && self.upper.is_infinite()
            && self.upper.is_sign_positive()
    }

    pub fn project(&self, val: f64) -> f64 {
        if self.lower > self.upper {
            return self.lower;
        }
        val.clamp(self.lower, self.upper)
    }

    pub fn violation(&self, val: f64) -> f64 {
        let lower_viol = if val < self.lower {
            self.lower - val
        } else {
            0.0
        };
        let upper_viol = if val > self.upper {
            val - self.upper
        } else {
            0.0
        };
        lower_viol.max(upper_viol)
    }
}

impl Default for Bound {
    fn default() -> Self {
        Self::non_negative()
    }
}
