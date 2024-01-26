#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeatureKind {
    Internal,
    Experiment,
}

pub type FeatureGate = (bool, FeatureKind);
