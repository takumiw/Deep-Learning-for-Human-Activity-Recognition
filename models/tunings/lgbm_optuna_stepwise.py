# -*- coding:utf-8 -*-
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import optuna
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in stderr.
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import log_loss, mean_squared_error
import lightgbm as lgb


class OptunaStepwise():
    n_splits = 5
    seed_cv = 71
    seed_optuna = 0
    seed_lgbm = 42
    learning_rate = 0.05

    def __init__(self, X_train, y_train, objective='multiclass', num_class=1, model_selection='StratifiedKFold', metric='log_loss', metric_lgbm='multi_logloss'):
        """
        Args:
            X_train (array): Explanatory variables in train dataset
            y_train (array): Target variable in train dataset
            objective (str): Objective for LightGBM ['binary', 'multiclass', 'regression']
            num_class (int): Number of classes
            model_selection (str): Cross validation methodology ['StratifiedKFold', 'KFold', 'GroupKFold']
            metric (str): Metric for evaluating cross validation score ['log_loss', 'mean_squared_error']
            metric_lgbm (str): Metric for early-stopping training LightGBM
        """
        self.X_train = X_train
        self.y_train = y_train
        
        if model_selection == 'StratifiedKFold':
            self.model_selection = StratifiedKFold
            self.cv = StratifiedKFold(n_splits=self.__class__.n_splits, shuffle=True, random_state=self.__class__.seed_cv)
        if metric == 'log_loss':
            self.metric = log_loss
        
        self.base_params = {
            "boosting_type": "gbdt",
            "objective": objective,
            "num_class": num_class,
            "metric": metric_lgbm,
            "learning_rate": self.__class__.learning_rate,
            "verbose": -1,
            "nthread": -1,
            "seed": self.__class__.seed_lgbm
        }


    def cross_validate(self, params: dict) -> float:
        """
        Cross validate for hyperparamter tuning.
        Args:
            params (dict): Hyperparameters for LightGBM
        Returns:
            retval (float): Mean of cross Validation Score
        """
        score_list = []
        for fold_id, (train_index, valid_index) in enumerate(self.cv.split(self.X_train, self.y_train)):
            X_tr = self.X_train.loc[train_index, :]
            X_val = self.X_train.loc[valid_index, :]
            y_tr = self.y_train.loc[train_index]
            y_val = self.y_train.loc[valid_index]

            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            model = lgb.train(
                    params, lgb_train,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=False,
                    num_boost_round=50000,
                    early_stopping_rounds=50)

            pred_val = model.predict(X_val, num_iteration=model.best_iteration)
            val_score = self.metric(y_val, pred_val)
            score_list.append(val_score)

        return np.mean(score_list)


    def obj1(self, trial):
        """
         1. Tuning num_leaves, max_depth
        num_leaves: default 31
        max_depth: default -1
        """
        params = {
            "num_leaves": trial.suggest_int('num_leaves', 15, 255),  
        }
        params["max_depth"] = trial.suggest_int('max_depth', np.ceil(np.log2(params['num_leaves'])), 20)
        params.update(self.base_params)
        return self.cross_validate(params)
    

    def obj2(self, trial):
        """
        2. TUning min_data_in_leaf, min_sum_hessian_in_leaf
        min_data_in_leaf: default 20, min_data_in_leaf >= 0
        min_sum_hessian_in_leaf: default 1e-3, min_sum_hessian_in_leaf >= 0.0
        """
        params = {
            "min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 5, 50),
            "min_sum_hessian_in_leaf": trial.suggest_loguniform('min_sum_hessian_in_leaf', 1e-4, 1e-2)
        }
        params.update(self.base_params)
        return self.cross_validate(params)


    def obj3(self, trial):
        """
        3. Tuning bagging_fraction, bagging_freq
        bagging_fraction: default 1.0, 0.0 < bagging_fraction <= 1.0
        bagging_freq: default 0
        """
        params = {
            "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            "bagging_freq": trial.suggest_int('bagging_freq', 0, 5)
        }
        params.update(self.base_params)
        return self.cross_validate(params)


    def obj4(self, trial):
        """
        4. Tuning reg_alpha, reg_lambda
        reg_alpha: default 0.0, reg_alpha >= 0.0
        reg_lambda: default 0.0, reg_lambda >= 0.0
        """
        params = {
            "reg_alpha": trial.suggest_uniform('reg_alpha', 0.0, 3.0),
            "reg_lambda": trial.suggest_uniform('reg_lambda', 0.0, 3.0)
        }
        params.update(self.base_params)
        return self.cross_validate(params)
    

    def obj5(self, trial):
        """
        5. Tuning max_bin
        max_bin: default 255, max_bin > 1
        """
        params = {
            "max_bin": trial.suggest_int('max_bin', 100, 500),
        }
        params.update(self.base_params)
        return self.cross_validate(params)

    
    def obj6(self, trial):
        """
        6. Tuning feature_fraction
        feature_fraction: default 1.0, 0.0 < feature_fraction <= 1.0
        """
        params = {
            "feature_fraction": trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        }
        params.update(self.base_params)
        return self.cross_validate(params)


    def start_tuning(self):
        
        # Create log file
        EXEC_TIME = 'lgbm-optuna-stepwise-' + datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.basicConfig(filename=f'../../logs/{EXEC_TIME}.log', level=logging.DEBUG)
        logging.debug(f'../../logs/{EXEC_TIME}.log')

        print(f'X_train:{self.X_train.shape} y_train:{self.y_train.shape}')
        logging.debug(f'X_train:{self.X_train.shape} y_train:{self.y_train.shape}')

        # ---以下、重要なパラメータから順にチューニングしていく---s
        for i, (obj, n_trials) in enumerate(zip([self.obj1,self.obj2,self.obj3,self.obj4,self.obj5,self.obj6], [40,40,40,40,30,30])):
            print(f'Start optimization {i+1}.')
            logging.getLogger().info(f'Start optimization {i+1}.')

            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=self.__class__.seed_optuna))
            study.optimize(obj, n_trials=n_trials)
            print(study.best_params)
            logging.debug(f'best params: {study.best_params}')

            self.base_params.update(study.best_params)
            print(f'Finished optimization {i+1}.\n{self.base_params}')
            logging.debug(f'Finished optimization {i+1}.\n{self.base_params}')