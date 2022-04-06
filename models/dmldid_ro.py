import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from IPython.display import clear_output
from scipy.stats import norm

class DMLDiD_RO:
    """
    hogehoge
    """

    def __init__(
        self,
        d_model=LogisticRegressionCV(
            cv=5, random_state=333, penalty="l1", solver="saga"
        ),
        l1k_model=LassoCV(cv=5, random_state=333),
        **kwargs,
    ):
        # params
        self._att_list = []
        self._att = None
        # model
        self.d_model = d_model
        self.l1k_model = l1k_model

    def fit(
        self,
        df:pd.DataFrame,
        y1_col:str,
        y0_col:str,
        d_col:str,
        X_cols:list,
        dmldid=True,
        sim_cnt=1,
        eps=0.03,
        base_random_seed=0,
        progress_plot=False,
        **kwargs,
    ):
        K = 2  # ２分割
        self._att_list = []  # 初期化
        for l in range(sim_cnt):
            if progress_plot & (l > 0):
                clear_output(wait=True)
                print(f"{l}. att : ", np.mean(self._att_list))
            if dmldid:
                df_set = train_test_split(
                    df, random_state=base_random_seed + l, test_size=0.5
                )
                temp_att = []
                for i in range(K):
                    k = 0 if i == 0 else 1
                    c = 1 if i == 0 else 0

                    self.d_model.fit(df_set[c][X_cols], df_set[c][d_col])

                    ghat = np.clip(
                        self.d_model.predict_proba(df_set[k][X_cols])[:, 1],
                        eps,
                        1 - eps,
                    )

                    control_y0 = df_set[c].query(f"{d_col} < 1")[y0_col]
                    control_y1 = df_set[c].query(f"{d_col} < 1")[y1_col]
                    _y = control_y1 - control_y0
                    control_x = df_set[c].query(f"{d_col} < 1")[X_cols]

                    self.l1k_model.fit(control_x, _y)
                    l1hat = self.l1k_model.predict(df_set[k][X_cols])

                    p_hat = df_set[k][d_col].mean()

                    _att = (
                        (df_set[k][y1_col] - df_set[k][y0_col] - l1hat)
                        / p_hat
                        * (df_set[k][d_col] - ghat)
                        / (1 - ghat)
                    ).mean()

                    temp_att.append(_att)
                self._att_list.append(np.mean(temp_att))
            else:
                # Abadie (2005)
                self.d_model.fit(df[X_cols], df[d_col])
                ghat = np.clip(
                    self.d_model.predict_proba(df[X_cols])[:, 1],
                    eps,
                    1 - eps,
                )

                p_hat = df[d_col].mean()
                self._att_list.append(
                    (
                        (df[y1_col] - df[y0_col])
                        / p_hat
                        * (df[d_col] - ghat)
                        / (1 - ghat)
                    ).mean()
                )

    def att(self):
        return np.mean(self._att_list)
    
    def sim_att_result(self):
        return self._att_list
    
    def summary(self, alpha=0.05):
        if len(self._att_list) < 2:
            return self.att()
        else:
            stderr = np.std(self._att_list)
            _effect = self.att()
            
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
