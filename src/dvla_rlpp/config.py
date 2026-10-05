"""Hyper-parameters of DVLA-RL++ (defaults follow the paper, Section 5.1)."""
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass
class PurificationConfig:
    """Complementary semantic purification (Section 3.3)."""
    tau_b: float = 0.1            # bank-response temperature in the size-normalised log-sum-exp, Eq. (4)
    tau_s: float = 0.2            # allocation temperature, Eq. (6)
    lam: float = 0.7              # visual-anchor coefficient lambda, Eq. (8)
    xi0: float = 0.05             # base rejection margin xi_0, Eq. (5)
    kappa_o: float = 0.1          # overlap slope kappa_o, Eq. (5)
    eps_n: float = 1e-3           # numerical fallback threshold, Eq. (9)
    tau_a: float = 0.1            # calibration temperature of the scoring maps, Eq. (S4)
    max_local: int = 8            # R, local phrases per class and role
    # Fixed local-to-global level schedule omega_{l,h} = (local, global)
    level_weights: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "stage2": (0.7, 0.3), "stage3": (0.3, 0.7), "final": (0.5, 0.5)})
    # Ablation switches (Table 5)
    uniform_allocation: bool = False   # a_i = 1 / M
    positive_only: bool = False        # drop the nuisance bank
    no_neutral: bool = False           # force 1^T a = 1
    fixed_margin: bool = False         # kappa_o = 0
    no_intrinsic_fallback: bool = False  # replace the anchor by mean visual evidence


@dataclass
class PolicyConfig:
    """Counterfactual reinforcement-learning gating (Section 3.4)."""
    kappa_g: float = 10.0         # Beta concentration
    eps_g: float = 0.05           # mean bounded in [eps_g, 1 - eps_g], Eq. (S5)
    hidden: int = 128
    gamma: float = 0.5            # nuisance-cost weight
    eta: float = 0.1              # reference-deviation weight
    eps_p: float = 0.2            # PPO clipping radius
    ppo_epochs: int = 4
    ppo_minibatch: int = 16       # episodes per surrogate mini-batch
    policy_lr: float = 1e-4
    value_lr: float = 1e-3
    gate: str = "policy"          # policy | reference | static  (static = support-independent scalar gate)
    state_baseline_only: bool = False  # ablation: advantage uses R - b instead of R - R0 - b


@dataclass
class ModelConfig:
    backbone: str = "visformer-t"
    image_size: int = 224
    text_dim: int = 512
    tau_c: float = 0.2            # classification temperature
    inject: Dict[str, int] = field(default_factory=lambda: {"stage2": 3, "stage3": 0})  # block index per stage
    purification: PurificationConfig = field(default_factory=PurificationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    use_csp: bool = True          # ablation: False -> uniform visual support prototypes
    use_cfg: bool = True          # ablation: False -> frozen reference gate only


def to_dict(cfg) -> dict:
    return asdict(cfg)
