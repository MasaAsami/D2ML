import os
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
from tqdm import tqdm

from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier, LGBMRegressor


class D2ML:
    """
    Doblue/Debiased Machine Learning
    """

    def __init__(
        self,
        data: pd.DataFrame,
        t_col: str,
        y_col: str,
        x_cols_for_t: list,
        x_cols_for_y: list,
        x_cols_for_cate: list,
        unit_id=None,
        t_model=LGBMClassifier(random_state=0),
        outcome_model=LGBMRegressor(random_state=0),
        cate_model=LGBMRegressor(random_state=0),
        n_bootstrap_samples=100,
    ):
        # params
        self._ate = None
        self._att = None
        self._bootstrap_ate = []
        self._bootstrap_att = []
        self.df_dml = None
        self.data = data
        self.t_col = t_col
        self.y_col = y_col
        self.x_cols_for_t = x_cols_for_t
        self.x_cols_for_y = x_cols_for_y
        self.x_cols_for_cate = x_cols_for_cate
        self.unit_id = unit_id
        self.n_bootstrap_samples = n_bootstrap_samples
        # model
        self.t_model = t_model
        self.outcome_model = outcome_model
        self.cate_model = cate_model
        self.ate_model_result = None
        self.explainer = None

    def fit_ate(
        self, eps=1e-3, cv=5, return_ate=False, bootstrap_data=pd.DataFrame([])
    ):
        if len(bootstrap_data) < 1:
            df_dml = self.data.copy()
        else:
            df_dml = bootstrap_data.copy()

        # t model for T residual
        df_dml = df_dml.assign(
            t_pred=np.clip(
                cross_val_predict(
                    self.t_model,
                    df_dml[self.x_cols_for_t],
                    df_dml[self.t_col],
                    cv=cv,
                    method="predict_proba",
                )[:, 1],
                eps,
                1 - eps,
            )
        )
        df_dml = df_dml.assign(
            t_res=df_dml[self.t_col] - df_dml["t_pred"],  # T residual
        )

        # y model for Y residual
        df_dml = df_dml.assign(
            y_res=df_dml[self.y_col]
            - cross_val_predict(
                self.outcome_model, df_dml[self.x_cols_for_y], df_dml[self.y_col], cv=cv
            ),
        )

        if self.unit_id == None:
            final_model = smf.ols(formula="y_res ~ t_res", data=df_dml).fit()
        else:
            final_model = smf.ols(formula="y_res ~ t_res", data=df_dml).fit(
                cov_type="cluster", cov_kwds={"groups": df_dml[self.unit_id]}
            )

        if return_ate:
            return final_model.params["t_res"], df_dml
        else:
            self._ate = final_model.params["t_res"]
            df_dml["ate"] = self._ate
            self.ate_model_result = final_model
            self.df_dml = df_dml

    def sim_bootstrap(self, eps=1e-3, cv=5, result_return=False):
        self._bootstrap_ate = []
        self._bootstrap_att = []
        _n = len(self.data)
        for i in tqdm(range(self.n_bootstrap_samples)):
            _df_dml = self.data.sample(n=_n, replace=True, random_state=i).reset_index(
                drop=True
            )
            with redirect_stdout(open(os.devnull, "w")):
                _ate, _df_dml = self.fit_ate(
                    eps=eps, cv=cv, return_ate=True, bootstrap_data=_df_dml
                )

            self._bootstrap_ate.append(_ate)

            _att = self.fit_nonlinear_cate(return_att=True, bootstrap_data=_df_dml)
            self._bootstrap_att.append(_att)

        self._att = np.mean(self._bootstrap_att)
        self._ate = np.mean(self._bootstrap_ate)  # rewrite ate
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
        if self._ate == None:
            self.fit_ate()

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

    def fit_linear_cate(self, linear_cate_Xcols=None):
        if self._ate == None:
            self.fit_ate()

        if linear_cate_Xcols == None:
            linear_cate_Xcols = self.x_cols_for_cate

        formula = "y_res ~ t_res*("
        formula += "+".join(linear_cate_Xcols)
        formula += ")"

        if self.unit_id == None:
            cate_linear_model = smf.ols(formula, data=self.df_dml).fit()
        else:
            cate_linear_model = smf.ols(formula, data=self.df_dml).fit(
                cov_type="cluster", cov_kwds={"groups": self.df_dml[self.unit_id]}
            )
        print("Warning: This STANDARD ERROR is not accurate!")
        print(cate_linear_model.summary().tables[1])

        self.df_dml = self.df_dml.assign(
            linear_cate=cate_linear_model.predict(self.df_dml.assign(t_res=1))
            - cate_linear_model.predict(self.df_dml.assign(t_res=0)),
        )

    def fit_nonlinear_cate(
        self, cap_att=True, return_att=False, bootstrap_data=pd.DataFrame([])
    ):
        if self._ate == None:
            self.fit_ate()

        if len(bootstrap_data) == 0:
            df_dml = self.df_dml.copy()
        else:
            df_dml = bootstrap_data.copy()

        model_weight = df_dml["t_res"] ** 2  # 残差の二乗
        cate_model_y = df_dml["y_res"] / df_dml["t_res"]  # 新しい教師ラベル

        self.cate_model.fit(
            X=df_dml[self.x_cols_for_cate], y=cate_model_y, sample_weight=model_weight
        )

        df_dml = df_dml.assign(
            non_linear_cate=self.cate_model.predict(df_dml[self.x_cols_for_cate])
        )
        # df_dml = df_dml.assign(
        #     non_linear_cate= cross_val_predict(
        #         self.cate_model, df_dml[self.x_cols_for_cate], cate_model_y, cv=cv, fit_params={"sample_weight":model_weight}
        #         )
        #     )
        if cap_att:
            q_low = df_dml["non_linear_cate"].quantile(0.01)
            q_hi = df_dml["non_linear_cate"].quantile(0.99)
            df_dml["non_linear_cate"] = df_dml["non_linear_cate"].clip(
                lower=q_low, upper=q_hi
            )

        if return_att:
            return df_dml.query(f"{self.t_col} > 0")["non_linear_cate"].mean()
        else:
            self.df_dml = df_dml
