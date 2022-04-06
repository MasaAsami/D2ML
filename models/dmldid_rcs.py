import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from IPython.display import clear_output

class DMLDiD_RCS:
    """
    hogehoge
    """

    def __init__(
        self,
        d_model=LogisticRegressionCV(
            cv=5, random_state=333, penalty="l1", solver="saga"
        ),
        l2k_model=LassoCV(cv=5, random_state=333),
        **kwargs,
    ):
        # params
        self._att_list = []
        self._att = None
        # model
        self.d_model = d_model
        self.l2k_model = l2k_model

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str,
        d_col: str,
        t_col: str,
        X_cols: list,
        dmldid=True,
        sim_cnt=1,
        eps=0.03,
        base_random_seed=0,
        progress_plot=False,
        d_model_t0_only=True,
        l2k_ps_weight=False,
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
                    df,
                    random_state=base_random_seed + l,
                    test_size=0.5,
                    stratify=df[[t_col, d_col]],
                )
                temp_att = []
                for i in range(K):
                    k = 0 if i == 0 else 1
                    c = 1 if i == 0 else 0

                    if d_model_t0_only:
                        self.d_model.fit(df_set[c].query(f"{t_col}<1")[X_cols], df_set[c].query(f"{t_col}<1")[d_col])
                    else:
                        self.d_model.fit(df_set[c][X_cols], df_set[c][d_col])

                    ghat = np.clip(
                        self.d_model.predict_proba(df_set[k][X_cols])[:, 1],
                        eps,
                        1 - eps,
                    )

                    lamda_hat = df_set[k][t_col].mean()
                    p_hat = df_set[k][d_col].mean()

                    # l2kmodel for T=1 and T=0 for control
                    l2kmodel_t1_c = self.l2k_model
                    l2kmodel_t0_c = self.l2k_model

                    control_y = df_set[c].query(f"{d_col} < 1")[y_col]
                    control_x = df_set[c].query(f"{d_col} < 1")[X_cols]
                    t_index = df_set[c].query(f"{d_col} < 1 & {t_col} > 0").index
                    not_t_index = df_set[c].query(f"{d_col} < 1 & {t_col} < 1").index

                    if l2k_ps_weight:
                        ps = np.clip(
                            self.d_model.predict_proba(control_x.loc[t_index])[:, 1],
                            eps,
                            1 - eps,
                        )
                        l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index], sample_weight = ps/(1-ps))
                        ps = np.clip(
                            self.d_model.predict_proba(control_x.loc[not_t_index])[:, 1],
                            eps,
                            1 - eps,
                        )
                        l2kmodel_t0_c.fit(
                            control_x.loc[not_t_index], control_y.loc[not_t_index], sample_weight= ps/(1-ps)
                        )
                    else:
                        l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index])
                        l2kmodel_t0_c.fit(
                            control_x.loc[not_t_index], control_y.loc[not_t_index]
                        )

                    l2k_hat_control_pre = l2kmodel_t0_c.predict(df_set[k][X_cols])
                    l2k_hat_control_post = l2kmodel_t1_c.predict(df_set[k][X_cols])

                    l2k_hat_control = l2k_hat_control_post * df_set[k][t_col] + l2k_hat_control_pre * (1 - df_set[k][t_col])

                    # l2kmodel for T=1 and T=0 for treated
                    l2kmodel_t1_t = self.l2k_model
                    l2kmodel_t0_t = self.l2k_model

                    treat_y = df_set[c].query(f"{d_col} > 0")[y_col]
                    treat_x = df_set[c].query(f"{d_col} > 0")[X_cols]
                    t_index = df_set[c].query(f"{d_col} > 0 & {t_col} > 0").index
                    not_t_index = df_set[c].query(f"{d_col} > 0 & {t_col} < 1").index

                    l2kmodel_t1_t.fit(treat_x.loc[t_index], treat_y.loc[t_index])
                    l2kmodel_t0_t.fit(
                        treat_x.loc[not_t_index], treat_y.loc[not_t_index]
                    )
                    
                    l2k_hat_treat_pre = l2kmodel_t0_t.predict(df_set[k][X_cols])
                    l2k_hat_treat_post = l2kmodel_t1_t.predict(df_set[k][X_cols])
                    # --------
                    # 1st stage
                    outcome_estimated_diff = df_set[k][y_col] - l2k_hat_control

                    _att = (
                        (df_set[k][t_col] - lamda_hat)
                        * outcome_estimated_diff
                        * (df_set[k][d_col] - ghat)
                        / ((1 - ghat) * lamda_hat * (1 - lamda_hat) * p_hat)
                    ).mean()
                    
                    # ## treat post & pre
                    # w_treat_post = df_set[k][t_col]*df_set[k][d_col]
                    # w_treat_pre = df_set[k][t_col]*df_set[k][d_col]
                    # treat_post_diff = w_treat_post*outcome_estimated_diff/(w_treat_post.mean())
                    # treat_pre_diff = w_treat_pre*outcome_estimated_diff/(w_treat_pre.mean())

                    # # control post & pre
                    # w_control_post = df_set[k][t_col]*(1 - df_set[k][d_col])*ghat/(1-ghat)
                    # w_control_pre = (1 - df_set[k][t_col])*(1 - df_set[k][d_col])*ghat/(1-ghat)

                    # control_post_diff = w_control_post*outcome_estimated_diff/(w_control_post.mean())
                    # control_pre_diff = w_control_pre*outcome_estimated_diff/(w_control_pre.mean())

                    # _att = (
                    #     (treat_post_diff - treat_pre_diff)
                    #     - (control_post_diff - control_pre_diff)
                    # ).mean()
                    
                    # --------
                    # 2nd stage
                    #  w.d * (out.y.treat.post - out.y.cont.post)/mean(w.d)
                    estimated_diff_post_d = df_set[k][d_col] * (l2k_hat_treat_post - l2k_hat_control_post)/p_hat
                    # w.dt1 * (out.y.treat.post - out.y.cont.post)/mean(w.dt1)
                    w_dt1 = df_set[k][d_col] * df_set[k][t_col]
                    estimated_diff_post_dt1 = w_dt1 * (l2k_hat_treat_post - l2k_hat_control_post)/(w_dt1.mean())
                    # w.d * (out.y.treat.pre - out.y.cont.pre)/mean(w.d)
                    estimated_diff_pre_d = df_set[k][d_col] * (l2k_hat_treat_pre - l2k_hat_control_pre)/p_hat
                    # eta.dt0.pre <- w.dt0 * (out.y.treat.pre - out.y.cont.pre)/mean(w.dt0)
                    w_dt0 = df_set[k][d_col] * (1 - df_set[k][t_col])
                    estimated_diff_pre_dt0 = w_dt0 * (l2k_hat_treat_pre - l2k_hat_control_pre)/(w_dt0.mean())

                    _att += (
                        (estimated_diff_post_d - estimated_diff_post_dt1)
                        - (estimated_diff_pre_d -  estimated_diff_pre_dt0)
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
                lamda_hat = df[t_col].mean()

                self._att_list.append(
                    (
                        (df[t_col] - lamda_hat)
                        * df[y_col]
                        * (df[d_col] - ghat)
                        / ((1 - ghat) * lamda_hat * (1 - lamda_hat) * p_hat)
                    ).mean()
                )

    def att(self):
        return np.mean(self._att_list)
    
    def sim_att_result(self):
        return self._att_list
