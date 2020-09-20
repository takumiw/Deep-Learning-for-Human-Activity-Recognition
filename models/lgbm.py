"""Functions for tarining Light GBM (LGBM)"""

from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from logs.logger_lgbm import log_evaluation

logger = getLogger(__name__)


def train_and_predict(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_valid: pd.DataFrame,
    lgbm_params: Dict[str, Any],
    verbose_eval: Union[int, bool] = 50,
    num_boost_round: int = 50000,
    early_stopping_rounds: int = 50,
    feval: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    callbacks = [log_evaluation(logger, period=verbose_eval)]

    model = lgb.train(
        lgbm_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=False,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
        feval=feval,
    )

    logger.debug(f"{model.best_iteration=}")
    logger.debug(f"train best score={model.best_score['training'].items()}")
    logger.debug(f"valid best score={model.best_score['valid_1'].items()}")

    pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)
    return pred_train, pred_valid, pred_test, model
