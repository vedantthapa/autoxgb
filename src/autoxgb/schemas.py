from typing import List, Optional

from pydantic import BaseModel

from .enums import ProblemType

import pandas as pd


class ModelConfig(BaseModel):
    train_file: pd.DataFrame
    test_file: Optional[pd.DataFrame] = None
    idx: str
    targets: List[str]
    problem_type: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]
    num_trials: int
    time_limit: Optional[int] = None
    fast: bool

    class Config:
        arbitrary_types_allowed = True
