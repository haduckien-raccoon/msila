"""Training utilities for MS-ILA Day-3."""

from .optimizer import (
    OptimizerContractError,
    OptimizerSetupReport,
    ParameterGroupSummary,
    assert_optimizer_contract,
    build_day3_msila_optimizer,
    build_optimizer,
    optimizer_parameter_ids,
)

"""MS-ILA training utilities."""

from .overfit16 import (
    EvaluationSummary,
    Overfit16Dataset,
    Overfit16DatasetError,
    Overfit16Trainer,
    OverfitTrainResult,
    OverfitTrainerError,
    StepLog,
    make_overfit16_loader,
    seed_everything,
)

__all__ = [
    "OptimizerContractError",
    "OptimizerSetupReport",
    "ParameterGroupSummary",
    "assert_optimizer_contract",
    "build_day3_msila_optimizer",
    "build_optimizer",
    "optimizer_parameter_ids",
    
    "EvaluationSummary",
    "Overfit16Dataset",
    "Overfit16DatasetError",
    "Overfit16Trainer",
    "OverfitTrainResult",
    "OverfitTrainerError",
    "StepLog",
    "make_overfit16_loader",
    "seed_everything",
]

