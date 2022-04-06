import os
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone


from lightgbm import LGBMClassifier, LGBMRegressor


class BaseLearner(object):
    def __init__(
        self,
        data: pd.DataFrame,
        t_col: str,
        y_col: str,
        x_cols: list,
        t_model=LGBMClassifier(random_state=0),
        outcome_model=LGBMRegressor(random_state=0),
        n_bootstrap_samples=100,
    ):
        # params
        self._ate = None
        self._att = None
        self._bootstrap_ate = []
        self._bootstrap_att = []
        self._df = None
        self.data = data
        self.t_col = t_col
        self.y_col = y_col
        self.x_cols = x_cols
        self.n_bootstrap_samples = n_bootstrap_samples
        # model
        self.t_model = t_model
        self.outcome_model = outcome_model
        self.ate_model_result = None

    def fit(self):
        pass

    def sim_bootstrap(self, eps=1e-3, result_return=False):
        self._bootstrap_ate = []
        self._bootstrap_att = []
        _n = len(self.data)
        for i in tqdm(range(self.n_bootstrap_samples)):
            _df = self.data.sample(n=_n, replace=True, random_state=i).reset_index(
                drop=True
            )
            with redirect_stdout(open(os.devnull, "w")):
                _df = self.fit(eps=eps, return_df=True, bootstrap_data=_df)

            self._bootstrap_ate.append(_df["cate"].mean())
            self._bootstrap_att.append(_df.query(f"{self.t_col}>0")["cate"].mean())

        self._ate = np.mean(self._bootstrap_ate)
        self._att = np.mean(self._bootstrap_att)
        if result_return:
            return self._bootstrap_ate, self._bootstrap_att

    def stderr(self, effect_type="ate"):
        if len(self._bootstrap_ate) == 0:
            self.sim_bootstrap()
        if effect_type == "ate":
            return np.std(self._bootstrap_ate)
        else:
            return np.std(self._bootstrap_att)

    def summary(self, alpha=0.05, effect_type="ate"):
        assert effect_type in ["ate", "att"]

        stderr = self.stderr(effect_type=effect_type)

        if effect_type == "ate":
            _effect = self._ate
        else:
            _effect = self._att

        z_stat = (_effect - 0) / stderr
        p_value = norm.sf(np.abs(z_stat), loc=0, scale=1) * 2
        # confidence interval
        ci_lower = norm.ppf(alpha / 2, loc=_effect, scale=stderr)
        ci_higher = norm.ppf(1 - alpha / 2, loc=_effect, scale=stderr)
        return pd.DataFrame(
            {
                "effect": _effect,
                "standard_error": stderr,
                "z_stat": z_stat,
                "p_value": p_value,
                f"CI {alpha}": ci_lower,
                f"CI {1-alpha}": ci_higher,
            },
            index=["t"],
        )


class SLearner(BaseLearner):
    def __init__(
        self,
        data: pd.DataFrame,
        t_col: str,
        y_col: str,
        x_cols: list,
        t_model=LGBMClassifier(random_state=0),
        outcome_model=LGBMRegressor(random_state=0),
        n_bootstrap_samples=100,
    ):
        super().__init__(
            data, t_col, y_col, x_cols, t_model, outcome_model, n_bootstrap_samples
        )

    def fit(self, eps=None, return_df=False, bootstrap_data=pd.DataFrame([])):
        if len(bootstrap_data) < 1:
            _df = self.data.copy()
        else:
            _df = bootstrap_data.copy()

        self.outcome_model.fit(_df[self.x_cols + [self.t_col]], _df[self.y_col])

        _dft1 = _df.copy()
        _dft1[self.t_col] = 1

        _dft0 = _df.copy()
        _dft0[self.t_col] = 0
        _df = _df.assign(
            m1_pred=self.outcome_model.predict(_dft1[self.x_cols + [self.t_col]]),
            m0_pred=self.outcome_model.predict(_dft0[self.x_cols + [self.t_col]]),
        )
        _df["cate"] = _df.eval("m1_pred - m0_pred")
        if return_df:
            return _df
        else:
            self._df = _df


class TLearner(BaseLearner):
    def __init__(
        self,
        data: pd.DataFrame,
        t_col: str,
        y_col: str,
        x_cols: list,
        t_model=LGBMClassifier(random_state=0),
        outcome_model=LGBMRegressor(random_state=0),
        n_bootstrap_samples=100,
    ):
        super().__init__(
            data, t_col, y_col, x_cols, t_model, outcome_model, n_bootstrap_samples
        )

    def fit(self, eps=None, return_df=False, bootstrap_data=pd.DataFrame([])):
        if len(bootstrap_data) < 1:
            _df = self.data.copy()
        else:
            _df = bootstrap_data.copy()

        outcome_model_for_treated = clone(self.outcome_model)
        outcome_model_for_control = clone(self.outcome_model)

        _dft1 = _df.query(f"{self.t_col} >0 ")
        _dft0 = _df.query(f"{self.t_col} < 1")
        outcome_model_for_treated.fit(_dft1[self.x_cols], _dft1[self.y_col])
        outcome_model_for_control.fit(_dft0[self.x_cols], _dft0[self.y_col])

        _df = _df.assign(
            m1_pred=outcome_model_for_treated.predict(_df[self.x_cols]),
            m0_pred=outcome_model_for_control.predict(_df[self.x_cols]),
        )
        _df["cate"] = _df.eval("m1_pred - m0_pred")
        if return_df:
            return _df
        else:
            self._df = _df


class XLearner(BaseLearner):
    def __init__(
        self,
        data: pd.DataFrame,
        t_col: str,
        y_col: str,
        x_cols: list,
        t_model=LGBMClassifier(random_state=0),
        outcome_model=LGBMRegressor(random_state=0),
        n_bootstrap_samples=100,
    ):
        super().__init__(
            data, t_col, y_col, x_cols, t_model, outcome_model, n_bootstrap_samples
        )

    def fit(self, eps=1e-3, return_df=False, cv=5, bootstrap_data=pd.DataFrame([])):
        if len(bootstrap_data) < 1:
            _df = self.data.copy()
        else:
            _df = bootstrap_data.copy()

        outcome_model_for_treated = clone(self.outcome_model)
        outcome_model_for_control = clone(self.outcome_model)

        _dft1 = _df.query(f"{self.t_col} > 0")
        _dft0 = _df.query(f"{self.t_col} < 1")
        outcome_model_for_treated.fit(_dft1[self.x_cols], _dft1[self.y_col])
        outcome_model_for_control.fit(_dft0[self.x_cols], _dft0[self.y_col])

        # using "control" model
        _dft1 = _dft1.assign(
            y_res=_dft1[self.y_col]
            - outcome_model_for_control.predict(_dft1[self.x_cols])
        )
        # using "treated" model
        _dft0 = _dft0.assign(
            y_res=outcome_model_for_treated.predict(_dft0[self.x_cols])
            - _dft0[self.y_col]
        )

        outcome_model_for_treated_2nd = clone(self.outcome_model)
        outcome_model_for_control_2nd = clone(self.outcome_model)

        outcome_model_for_treated_2nd.fit(_dft1[self.x_cols], _dft1["y_res"])
        outcome_model_for_control_2nd.fit(_dft0[self.x_cols], _dft0["y_res"])

        _df = _df.assign(
            m1_pred=outcome_model_for_treated_2nd.predict(_df[self.x_cols]),
            m0_pred=outcome_model_for_control_2nd.predict(_df[self.x_cols]),
            ps=np.clip(
                cross_val_predict(
                    self.t_model,
                    _df[self.x_cols],
                    _df[self.t_col],
                    cv=cv,
                    method="predict_proba",
                )[:, 1],
                eps,
                1 - eps,
            ),
        )
        _df["cate"] = _df.eval("ps*m0_pred + (1-ps)*m1_pred")
        if return_df:
            return _df
        else:
            self._df = _df


class DomainAdaptationLearner(BaseLearner):
    def __init__(
        self,
        data: pd.DataFrame,
        t_col: str,
        y_col: str,
        x_cols: list,
        t_model=LGBMClassifier(random_state=0),
        outcome_model=LGBMRegressor(random_state=0),
        n_bootstrap_samples=100,
    ):
        super().__init__(
            data, t_col, y_col, x_cols, t_model, outcome_model, n_bootstrap_samples
        )

    def fit(self, eps=1e-3, return_df=False, cv=5, bootstrap_data=pd.DataFrame([])):
        if len(bootstrap_data) < 1:
            _df = self.data.copy()
        else:
            _df = bootstrap_data.copy()

        # propensity socre
        _df = _df.assign(
            ps=np.clip(
                cross_val_predict(
                    self.t_model,
                    _df[self.x_cols],
                    _df[self.t_col],
                    cv=cv,
                    method="predict_proba",
                )[:, 1],
                eps,
                1 - eps,
            ),
        )

        outcome_model_for_treated = clone(self.outcome_model)
        outcome_model_for_control = clone(self.outcome_model)

        _dft1 = _df.query(f"{self.t_col} > 0")
        _dft1["weight"] = _dft1.eval("(1-ps)/ps")
        _dft0 = _df.query(f"{self.t_col} < 1")
        _dft0["weight"] = _dft0.eval("ps/(1-ps)")

        outcome_model_for_treated.fit(
            _dft1[self.x_cols], _dft1[self.y_col], sample_weight=_dft1["weight"]
        )
        outcome_model_for_control.fit(
            _dft0[self.x_cols], _dft0[self.y_col], sample_weight=_dft0["weight"]
        )

        # using "control" model
        _dft1 = _dft1.assign(
            y_res=_dft1[self.y_col]
            - outcome_model_for_control.predict(_dft1[self.x_cols])
        )
        # using "treated" model
        _dft0 = _dft0.assign(
            y_res=outcome_model_for_treated.predict(_dft0[self.x_cols])
            - _dft0[self.y_col]
        )

        outcome_model_for_final = clone(self.outcome_model)

        _df = pd.concat([_dft0, _dft1])
        outcome_model_for_final.fit(_df[self.x_cols], _df["y_res"])

        _df["cate"] = outcome_model_for_final.predict(_df[self.x_cols])
        if return_df:
            return _df
        else:
            self._df = _df
