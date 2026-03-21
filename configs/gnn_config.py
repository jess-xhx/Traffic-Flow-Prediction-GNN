from dataclasses import dataclass


@dataclass
class ModelConfig:
    static_dim: int
    profile_dim: int
    event_dim: int
    static_hidden_dim: int = 32
    calendar_hidden_dim: int = 32
    profile_hidden_dim: int = 32
    bank_hidden_dim: int = 64
    recent_hidden_dim: int = 32


@dataclass
class StageTrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float | None = 5.0
    log_every: int = 1


@dataclass
class JointTrainConfig:
    epochs: int = 5
    lr_base: float = 1e-4
    lr_recent: float = 3e-4
    lr_event: float = 5e-4
    weight_decay: float = 1e-5
    alpha_base: float = 0.2
    beta_recent: float = 0.3
    gamma_event: float = 1.0
    lambda_recent_reg: float = 1e-4
    lambda_event_reg: float = 1e-4
    grad_clip: float | None = 5.0
    log_every: int = 1
