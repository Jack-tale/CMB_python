#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is adapted from the EPA Chemical Mass Balance (CMB) model (https://www.epa.gov/scram/chemical-mass-balance-cmb-model).


import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg
from tqdm import tqdm

ERR_ZERO_AMB_UNC = "ERROR_CMBSVD_2"
ERR_ZERO_COLUMN = "ERROR_CMBSVD_4"
ERR_SINGULAR = "ERROR_CMBSVD_5"
ERR_NONCONVERGENCE = "ERROR_CMBSVD_6"
ERR_NO_FIT_SPECIES = "ERROR_NO_FIT_SPECIES"
ERR_NO_FIT_SOURCES = "ERROR_NO_FIT_SOURCES"
ERR_DIMENSION = "ERROR_DIMENSION"
ERR_NUMERIC = "ERROR_NUMERIC"
ERR_INPUT = "ERROR_INPUT"
ERR_SVD = "ERROR_SVD"
ERR_UNEXPECTED = "UNEXPECTED_ERROR"


from dataclasses import dataclass, asdict


@dataclass
class CMBConfig:
    miss_value: float = -99.0
    tol: float = 1.0e-5
    convergence_threshold: float = 0.01
    max_source_unc_pct: float = 100.0
    min_source_proj: float = 0.1
    max_iterations: int = 100
    use_britt_luecke: bool = False
    auto_eliminate_negative: bool = True
    source_unc_zero_replacement: float = 1.0e-5
    verbose: bool = False


class CMBFitError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def read_table_auto(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    elif ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        if sheet_name is None:
            return pd.read_excel(path, sheet_name=0)
        return pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f"不支持的文件格式: {path}")


class CMBModel:
    def __init__(self, config: Optional[CMBConfig] = None):
        self.config = config or CMBConfig()

        self.source_names: List[str] = []
        self.species_names: List[str] = []
        self.n_sources: int = 0
        self.n_species: int = 0

        self.atotal: Optional[np.ndarray] = None
        self.uatotal: Optional[np.ndarray] = None

        self.amb_df: Optional[pd.DataFrame] = None
        self.amb_unc_df: Optional[pd.DataFrame] = None

        self.source_unc_zero_count: int = 0

        self.current_sample: Optional[str] = None
        self.current_row_index: Optional[int] = None

        self.amb_sp: Optional[np.ndarray] = None
        self.u_amb_sp: Optional[np.ndarray] = None

        self.i_sp_fit: List[int] = []
        self.j_pr_fit: List[int] = []
        self.current_fit_species_indices: List[int] = []
        self.current_fit_source_indices: List[int] = []
        self.current_excluded_species_indices: List[int] = []

        self.n_sp_fit: int = 0
        self.n_pr_fit: int = 0

        self.sc_e: Optional[np.ndarray] = None
        self.u_sc_e: Optional[np.ndarray] = None
        self.c_hat: Optional[np.ndarray] = None
        self.u_c_hat: Optional[np.ndarray] = None
        self.sng_val: Optional[np.ndarray] = None
        self.egn_vec: Optional[np.ndarray] = None
        self.u_array: Optional[np.ndarray] = None
        self.m_pin: Optional[np.ndarray] = None

        self.rsquare: float = np.nan
        self.chi_sqr: float = np.nan
        self.deg_free: int = 0
        self.pcnt_mass: float = np.nan
        self.sum_sce: float = np.nan

        self.eligible_dim: int = 0
        self.projections: Optional[np.ndarray] = None
        self.num_est_sources: int = 0
        self.frac_est: float = 0.0


    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def _algorithm_name(self) -> str:
        return "Britt-Luecke" if self.config.use_britt_luecke else "Standard"

    def _is_miss_value(self, x: float) -> bool:
        return abs(float(x) - self.config.miss_value) < self.config.tol

    def _is_miss_array(self, arr: np.ndarray) -> np.ndarray:
        return np.abs(arr - self.config.miss_value) < self.config.tol

    @staticmethod
    def _unique_preserve_order(seq: Sequence[Any]) -> List[Any]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _source_names_from_indices(self, indices: Sequence[int]) -> List[str]:
        return [self.source_names[i - 1] for i in indices]

    def _species_names_from_indices(self, indices: Sequence[int]) -> List[str]:
        return [self.species_names[i - 1] for i in indices]

    def _reset_current_fit_state(self) -> None:
        self.i_sp_fit = []
        self.j_pr_fit = []
        self.current_fit_species_indices = []
        self.current_fit_source_indices = []
        self.current_excluded_species_indices = []

        self.n_sp_fit = 0
        self.n_pr_fit = 0

        self.sc_e = None
        self.u_sc_e = None
        self.c_hat = None
        self.u_c_hat = None
        self.sng_val = None
        self.egn_vec = None
        self.u_array = None
        self.m_pin = None

        self.rsquare = np.nan
        self.chi_sqr = np.nan
        self.deg_free = 0
        self.pcnt_mass = np.nan
        self.sum_sce = np.nan

        self.eligible_dim = 0
        self.projections = None
        self.num_est_sources = 0
        self.frac_est = 0.0

    def load_data_from_files(
        self,
        source_profile_path: str,
        source_unc_path: str,
        amb_path: str,
        amb_unc_path: str,
        sheet_names: Optional[Dict[str, Optional[str]]] = None
    ) -> None:
        sheet_names = sheet_names or {}

        source_profile = read_table_auto(
            source_profile_path,
            sheet_names.get("source_profile")
        )
        source_unc = read_table_auto(
            source_unc_path,
            sheet_names.get("source_unc")
        )
        amb = read_table_auto(
            amb_path,
            sheet_names.get("amb")
        )
        amb_unc = read_table_auto(
            amb_unc_path,
            sheet_names.get("amb_unc")
        )

        self.load_data(source_profile, source_unc, amb, amb_unc)

    def load_data(
        self,
        source_profile: pd.DataFrame,
        source_unc: pd.DataFrame,
        amb: pd.DataFrame,
        amb_unc: pd.DataFrame
    ) -> None:
        for name, df in [
            ("source_profile", source_profile),
            ("source_unc", source_unc),
            ("amb", amb),
            ("amb_unc", amb_unc),
        ]:
            if df.shape[1] < 2:
                raise ValueError(f"{name} 至少应包含 2 列：第一列标签 + 至少一个物种列。")

        self.species_names = source_profile.columns[1:].tolist()

        for i, cols in enumerate(
            [
                source_unc.columns[1:].tolist(),
                amb.columns[1:].tolist(),
                amb_unc.columns[1:].tolist(),
            ],
            start=1
        ):
            if cols != self.species_names:
                raise ValueError(
                    f"第 {i} 个表的物种列名或顺序与 source_profile 不一致。\n"
                    f"期望: {self.species_names}\n"
                    f"实际: {cols}"
                )

        src_names_profile = source_profile.iloc[:, 0].astype(str).tolist()
        src_names_unc = source_unc.iloc[:, 0].astype(str).tolist()
        if src_names_profile != src_names_unc:
            raise ValueError("source_profile 与 source_unc 的第一列（源名称）顺序不一致。")

        amb_labels = amb.iloc[:, 0].astype(str).tolist()
        amb_unc_labels = amb_unc.iloc[:, 0].astype(str).tolist()
        if amb_labels != amb_unc_labels:
            raise ValueError("amb 与 amb_unc 的第一列（样品名称）顺序不一致。")

        for name, df in [
            ("source_profile", source_profile),
            ("source_unc", source_unc),
            ("amb", amb),
            ("amb_unc", amb_unc),
        ]:
            if df.isnull().any().any():
                raise ValueError(f"{name} 中存在 NaN，请先处理。")

        try:
            src_prof_num = source_profile.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
            src_unc_num = source_unc.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
            amb_num = amb.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
            amb_unc_num = amb_unc.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
        except Exception as e:
            raise ValueError("数值列中存在无法转换为数值的内容。") from e

        zero_mask = (src_unc_num == 0.0)
        self.source_unc_zero_count = int(zero_mask.sum().sum())

        if self.source_unc_zero_count > 0:
            self._log(
                f"检测到 source_unc 中有 {self.source_unc_zero_count} 个 0，"
                f"已自动替换为 {self.config.source_unc_zero_replacement:.1e}"
            )
            src_unc_num = src_unc_num.mask(
                zero_mask,
                self.config.source_unc_zero_replacement
            )

        self.source_names = src_names_profile
        self.n_sources = len(self.source_names)
        self.n_species = len(self.species_names)

        self.atotal = np.zeros((self.n_species + 1, self.n_sources + 1), dtype=np.float64)
        self.uatotal = np.zeros((self.n_species + 1, self.n_sources + 1), dtype=np.float64)

        self.atotal[1:, 1:] = src_prof_num.to_numpy(dtype=np.float64).T
        self.uatotal[1:, 1:] = src_unc_num.to_numpy(dtype=np.float64).T

        self.amb_df = amb.copy()
        self.amb_unc_df = amb_unc.copy()

        self._log("数据加载成功。")
        self._log(f"  物种数（含总质量）: {self.n_species}")
        self._log(f"  源数: {self.n_sources}")
        self._log(f"  环境样品数: {len(self.amb_df)}")

    def _normalize_species_selection(
        self,
        fit_species: Optional[Sequence[Any]] = None
    ) -> List[int]:
        if fit_species is None:
            return list(range(1, self.n_species + 1))

        out: List[int] = []
        for item in fit_species:
            if isinstance(item, str):
                if item not in self.species_names:
                    raise ValueError(f"未找到物种: {item}")
                out.append(self.species_names.index(item) + 1)
            else:
                idx = int(item)
                if idx < 1 or idx > self.n_species:
                    raise ValueError(f"物种索引越界: {idx}")
                out.append(idx)

        return self._unique_preserve_order(out)

    def _normalize_source_selection(
        self,
        fit_sources: Optional[Sequence[Any]] = None
    ) -> List[int]:
        if fit_sources is None:
            return list(range(1, self.n_sources + 1))

        out: List[int] = []
        for item in fit_sources:
            if isinstance(item, str):
                if item not in self.source_names:
                    raise ValueError(f"未找到源: {item}")
                out.append(self.source_names.index(item) + 1)
            else:
                idx = int(item)
                if idx < 1 or idx > self.n_sources:
                    raise ValueError(f"源索引越界: {idx}")
                out.append(idx)

        return self._unique_preserve_order(out)

    def _prepare_sample_vectors(self, row_pos: int) -> None:
        if self.amb_df is None or self.amb_unc_df is None:
            raise RuntimeError("尚未加载环境数据。")

        if row_pos < 0 or row_pos >= len(self.amb_df):
            raise IndexError(f"样品行号越界: {row_pos}")

        sample_label = str(self.amb_df.iat[row_pos, 0])

        amb_row = pd.to_numeric(
            self.amb_df.iloc[row_pos, 1:],
            errors="raise"
        ).to_numpy(dtype=np.float64)

        unc_row = pd.to_numeric(
            self.amb_unc_df.iloc[row_pos, 1:],
            errors="raise"
        ).to_numpy(dtype=np.float64)

        self.current_sample = sample_label
        self.current_row_index = row_pos

        self.amb_sp = np.zeros(self.n_species + 1, dtype=np.float64)
        self.u_amb_sp = np.zeros(self.n_species + 1, dtype=np.float64)

        self.amb_sp[1:] = amb_row
        self.u_amb_sp[1:] = unc_row

    def _set_fitting_elements_for_sample(
        self,
        active_sources: Sequence[int],
        candidate_species: Sequence[int]
    ) -> None:
        if self.amb_sp is None or self.u_amb_sp is None:
            raise RuntimeError("当前样品环境数据尚未准备。")

        self.j_pr_fit = list(active_sources)
        self.current_fit_source_indices = list(active_sources)
        self.current_excluded_species_indices = []

        if len(self.j_pr_fit) == 0:
            raise CMBFitError(ERR_NO_FIT_SOURCES, "当前样品没有可参与拟合的源。")

        self.i_sp_fit = []

        for i in candidate_species:
            if self._is_miss_value(self.amb_sp[i]):
                self.current_excluded_species_indices.append(i)
                continue

            profile_row = self.atotal[i, self.j_pr_fit]
            if np.any(self._is_miss_array(np.asarray(profile_row, dtype=np.float64))):
                self.current_excluded_species_indices.append(i)
                continue

            self.i_sp_fit.append(i)

        self.current_fit_species_indices = list(self.i_sp_fit)
        self.n_sp_fit = len(self.i_sp_fit)
        self.n_pr_fit = len(self.j_pr_fit)

        if self.n_sp_fit == 0:
            raise CMBFitError(
                ERR_NO_FIT_SPECIES,
                "剔除 MISS 后，当前样品没有可用于拟合的物种。"
            )

        if self.n_pr_fit == 0:
            raise CMBFitError(
                ERR_NO_FIT_SOURCES,
                "当前样品没有可用于拟合的源。"
            )

        if self.n_sp_fit < self.n_pr_fit:
            raise CMBFitError(
                ERR_DIMENSION,
                f"当前样品有效拟合物种数({self.n_sp_fit})小于当前拟合源数({self.n_pr_fit})。"
            )

        for i in self.i_sp_fit:
            if self.u_amb_sp[i] <= 0.0:
                raise CMBFitError(
                    ERR_ZERO_AMB_UNC,
                    f'拟合物种 "{self.species_names[i - 1]}" 的环境不确定度 <= 0。'
                )


    def _build_selected_matrices(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sp = np.asarray(self.i_sp_fit, dtype=int)
        src = np.asarray(self.j_pr_fit, dtype=int)

        Aorg = self.atotal[np.ix_(sp, src)].astype(np.float64, copy=True)
        UA = self.uatotal[np.ix_(sp, src)].astype(np.float64, copy=True)
        cvec = self.amb_sp[sp].astype(np.float64, copy=True)
        uamb = self.u_amb_sp[sp].astype(np.float64, copy=True)

        return sp, src, Aorg, UA, cvec, uamb

    def _check_initial_fit_errors(
        self,
        Aorg: np.ndarray,
        src_indices: np.ndarray
    ) -> None:
        zero_cols = np.where(np.all(np.abs(Aorg) <= self.config.tol, axis=0))[0]
        if len(zero_cols) > 0:
            bad_sources = [self.source_names[src_indices[j] - 1] for j in zero_cols]
            raise CMBFitError(
                ERR_ZERO_COLUMN,
                f"Afit 中存在全零列（源列全为 0），对应源: {bad_sources}"
            )

    def _compute_veff_inverse(
        self,
        uamb: np.ndarray,
        UA: np.ndarray,
        sfit: np.ndarray
    ) -> np.ndarray:
        veff_den = uamb ** 2 + np.sum((UA * sfit[np.newaxis, :]) ** 2, axis=1)

        if not np.all(np.isfinite(veff_den)):
            raise CMBFitError(ERR_NUMERIC, "有效方差分母出现 NaN 或 Inf。")

        if np.any(veff_den <= 0.0):
            raise CMBFitError(ERR_NUMERIC, "有效方差分母出现 <= 0。")

        veff_inv = 1.0 / veff_den

        if not np.all(np.isfinite(veff_inv)):
            raise CMBFitError(ERR_NUMERIC, "有效方差逆出现 NaN 或 Inf。")

        return veff_inv

    def _solve_weighted_system_cholesky(
        self,
        Afit: np.ndarray,
        veff_inv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ata = Afit.T @ (veff_inv[:, np.newaxis] * Afit)

        try:
            cho = linalg.cho_factor(ata, lower=False, check_finite=False)
            inv_ata = linalg.cho_solve(
                cho,
                np.eye(ata.shape[0], dtype=np.float64),
                check_finite=False
            )
        except linalg.LinAlgError as e:
            raise CMBFitError(
                ERR_SINGULAR,
                "A^T * Veff^-1 * A 无法进行 Cholesky 分解，通常意味着共线性或病态矩阵。"
            ) from e

        uarray_step = inv_ata @ (Afit.T * veff_inv[np.newaxis, :])

        if not np.all(np.isfinite(uarray_step)):
            raise CMBFitError(ERR_NUMERIC, "求解矩阵出现 NaN 或 Inf。")

        return inv_ata, uarray_step

    @staticmethod
    def _fortran_converged(
        old_s: np.ndarray,
        new_s: np.ndarray,
        threshold: float
    ) -> bool:
        for old_val, new_val in zip(old_s, new_s):
            if new_val != 0.0:
                ratio = (old_val - new_val) / new_val
                if abs(ratio) > threshold:
                    return False
        return True

    def _update_britt_luecke_afit(
        self,
        Aorg: np.ndarray,
        UA: np.ndarray,
        Afit_current: np.ndarray,
        uarray_step: np.ndarray,
        veff_inv: np.ndarray,
        cvec: np.ndarray,
        old_s: np.ndarray
    ) -> np.ndarray:
        n_sp, n_pr = Aorg.shape

        projection = Afit_current @ uarray_step

        middle = np.eye(n_sp, dtype=np.float64) - projection

        middle = veff_inv[:, np.newaxis] * middle

        residual_old = cvec - (Aorg @ old_s)

        work_vec = middle @ residual_old

        if not np.all(np.isfinite(work_vec)):
            raise CMBFitError(ERR_NUMERIC, "Britt-Luecke 更新过程中出现 NaN 或 Inf。")

        Afit_new = np.empty_like(Afit_current)

        for j in range(n_pr):
            # workix1 = (uAtot(:,j)^2) * work_vec
            # workix1 = Sold(j) * workix1
            # Afit(:,j) = Aorg(:,j) + workix1
            correction = old_s[j] * (UA[:, j] ** 2) * work_vec
            Afit_new[:, j] = Aorg[:, j] + correction

        if not np.all(np.isfinite(Afit_new)):
            raise CMBFitError(ERR_NUMERIC, "Britt-Luecke 更新后的 Afit 出现 NaN 或 Inf。")

        return Afit_new


    def _perform_cmb_fit(self) -> None:
        sp, src, Aorg, UA, cvec, uamb = self._build_selected_matrices()

        self._check_initial_fit_errors(Aorg, src)

        n_sp, n_pr = Aorg.shape

        sfit = np.zeros(n_pr, dtype=np.float64)

        Afit_iter = Aorg.copy()
        veff_inv_final = None

        converged = False

        for loop in range(1, self.config.max_iterations + 1):
            old_s = sfit.copy()

            veff_inv = self._compute_veff_inverse(uamb, UA, sfit)

            _, uarray_step = self._solve_weighted_system_cholesky(Afit_iter, veff_inv)

            if not self.config.use_britt_luecke:
                s_new = uarray_step @ cvec
            else:
                residual_old = cvec - (Aorg @ old_s)
                delta_s = uarray_step @ residual_old
                s_new = old_s + delta_s

            if not np.all(np.isfinite(s_new)):
                raise CMBFitError(ERR_NUMERIC, "源贡献解中出现 NaN 或 Inf。")

            sfit = s_new

            if self._fortran_converged(old_s, sfit, self.config.convergence_threshold):
                converged = True
                veff_inv_final = veff_inv
                break

            if loop < self.config.max_iterations:
                if self.config.use_britt_luecke:
                    Afit_iter = self._update_britt_luecke_afit(
                        Aorg=Aorg,
                        UA=UA,
                        Afit_current=Afit_iter,
                        uarray_step=uarray_step,
                        veff_inv=veff_inv,
                        cvec=cvec,
                        old_s=old_s
                    )
            else:
                break

        if not converged:
            raise CMBFitError(
                ERR_NONCONVERGENCE,
                f"在 {self.config.max_iterations} 次迭代内未收敛。"
            )

        self.sc_e = sfit.copy()

        self.c_hat = np.full(self.n_species + 1, self.config.miss_value, dtype=np.float64)
        self.u_c_hat = np.full(self.n_species + 1, self.config.miss_value, dtype=np.float64)

        for global_i in range(2, self.n_species + 1):
            arow = self.atotal[global_i, src]
            if np.any(self._is_miss_array(np.asarray(arow, dtype=np.float64))):
                continue

            urow = self.uatotal[global_i, src]

            self.c_hat[global_i] = float(np.dot(arow, sfit))
            self.u_c_hat[global_i] = float(np.sqrt(np.sum((urow * sfit) ** 2)))

        for local_i in range(1, n_sp):
            global_i = sp[local_i]
            self.c_hat[global_i] = float(np.dot(Afit_iter[local_i, :], sfit))


        Aw = Afit_iter * np.sqrt(veff_inv_final)[:, np.newaxis]

        if not np.all(np.isfinite(Aw)):
            raise CMBFitError(ERR_NUMERIC, "加权后的 Afit 出现 NaN 或 Inf。")

        U, svals, Vh = linalg.svd(Aw, full_matrices=False, check_finite=False)

        if not np.all(np.isfinite(svals)):
            raise CMBFitError(ERR_SVD, "奇异值分解结果出现 NaN 或 Inf。")

        if np.any(svals <= 0.0):
            raise CMBFitError(ERR_SVD, "奇异值中出现 <= 0，无法继续计算不确定度。")

        V = Vh.T

        self.sng_val = svals.copy()
        self.egn_vec = V.copy()
        self.u_array = U.copy()

        cov_s = V @ np.diag(1.0 / (svals ** 2)) @ V.T

        if not np.all(np.isfinite(cov_s)):
            raise CMBFitError(ERR_NUMERIC, "源贡献协方差矩阵出现 NaN 或 Inf。")

        self.u_sc_e = np.sqrt(np.maximum(np.diag(cov_s), 0.0))

        self.sum_sce = float(np.sum(sfit))
        self.c_hat[1] = self.sum_sce

        ones = np.ones(n_pr, dtype=np.float64)
        total_var = float(ones @ cov_s @ ones)
        self.u_c_hat[1] = float(np.sqrt(max(total_var, 0.0)))

        self.deg_free = n_sp - n_pr

        weighted_amb = np.sqrt(veff_inv_final) * cvec
        weighted_calc = np.sqrt(veff_inv_final) * self.c_hat[sp]
        weighted_resid = weighted_amb - weighted_calc

        x3 = float(np.sum(weighted_amb ** 2))
        x4 = float(np.sum(weighted_resid ** 2))

        self.chi_sqr = x4 / self.deg_free if self.deg_free > 0 else np.nan
        self.rsquare = 1.0 - x4 / x3 if x3 != 0.0 else np.nan

        if self.amb_sp is not None and not self._is_miss_value(self.amb_sp[1]) and self.amb_sp[1] != 0.0:
            self.pcnt_mass = 100.0 * self.c_hat[1] / self.amb_sp[1]
        else:
            self.pcnt_mass = np.nan

        self.m_pin = np.zeros((n_sp, n_pr), dtype=np.float64)

        for i in range(n_sp):
            for j in range(n_pr):
                self.m_pin[i, j] = float(np.sum(U[i, :] * V[j, :] / svals))

        for j in range(n_pr):
            col_max = np.max(np.abs(self.m_pin[:, j]))
            if col_max > 0.0:
                self.m_pin[:, j] /= col_max

    def _find_most_negative_source(self) -> Tuple[Optional[int], Optional[float]]:
        if self.sc_e is None or len(self.sc_e) == 0:
            return None, None

        idx = int(np.argmin(self.sc_e))
        val = float(self.sc_e[idx])

        if val < 0.0:
            return idx, val
        return None, None

    def _calculate_estimable(self) -> None:
        if self.sng_val is None or self.egn_vec is None:
            raise RuntimeError("SVD 结果不存在，无法计算 estimable source。")

        max_src_unc_abs = self.config.max_source_unc_pct * self.amb_sp[1] / 100.0

        self.eligible_dim = self.n_pr_fit
        for j in range(self.n_pr_fit):
            if self.sng_val[j] < 1.0e-10:
                self.eligible_dim = j
                break
            if (1.0 / self.sng_val[j]) > max_src_unc_abs:
                self.eligible_dim = j
                break

        self.projections = np.sqrt(
            np.sum(self.egn_vec[:, :self.eligible_dim] ** 2, axis=1)
        )

        self.num_est_sources = int(np.sum(self.projections >= self.config.min_source_proj))
        self.frac_est = (
            self.num_est_sources / self.n_pr_fit if self.n_pr_fit > 0 else 0.0
        )

    def _build_success_result(
        self,
        eliminated_sources: List[str]
    ) -> Dict[str, Any]:
        message_parts = ["求解成功"]

        if len(self.current_excluded_species_indices) > 0:
            message_parts.append(
                f"因 MISS 自动排除物种 {len(self.current_excluded_species_indices)} 个"
            )

        if len(eliminated_sources) > 0:
            message_parts.append("自动剔除负源: " + ", ".join(eliminated_sources))

        if self.n_pr_fit == 1 and self.sc_e is not None and self.sc_e[0] < 0.0:
            message_parts.append("仅剩 1 个源且为负值，按 Fortran 逻辑不再继续剔除")

        return {
            "row_index": self.current_row_index,
            "sample": self.current_sample,
            "status": "OK",
            "error_code": "",
            "message": "；".join(message_parts),
            "algorithm": self._algorithm_name(),
            "converged": True,

            "ambient_mass": float(self.amb_sp[1]) if self.amb_sp is not None else np.nan,
            "u_ambient_mass": float(self.u_amb_sp[1]) if self.u_amb_sp is not None else np.nan,

            "sum_sce": float(self.sum_sce),
            "rsquare": float(self.rsquare),
            "chi_sqr": float(self.chi_sqr) if np.isfinite(self.chi_sqr) else np.nan,
            "deg_free": int(self.deg_free),
            "pcnt_mass": float(self.pcnt_mass) if np.isfinite(self.pcnt_mass) else np.nan,

            "eligible_dim": int(self.eligible_dim),
            "num_est_sources": int(self.num_est_sources),
            "frac_est": float(self.frac_est),

            "fit_species_indices": list(self.current_fit_species_indices),
            "fit_species_names": self._species_names_from_indices(self.current_fit_species_indices),
            "excluded_species_indices": list(self.current_excluded_species_indices),
            "excluded_species_names": self._species_names_from_indices(self.current_excluded_species_indices),

            "fit_source_indices": list(self.current_fit_source_indices),
            "source_names": self._source_names_from_indices(self.current_fit_source_indices),

            "eliminated_sources": list(eliminated_sources),

            "sc_e": self.sc_e.copy(),
            "u_sc_e": self.u_sc_e.copy(),
            "singular_values": self.sng_val.copy(),
            "projections": self.projections.copy(),

            "species_names": list(self.species_names),
            "species_measured": self.amb_sp[1:].copy(),
            "species_unc": self.u_amb_sp[1:].copy(),
            "species_calculated": self.c_hat[1:].copy(),
            "species_calc_unc": self.u_c_hat[1:].copy(),

            "m_pin": self.m_pin.copy() if self.m_pin is not None else None,
            "m_pin_species_names": self._species_names_from_indices(self.current_fit_species_indices),
            "m_pin_source_names": self._source_names_from_indices(self.current_fit_source_indices),
        }

    def _build_failure_result(
        self,
        error_code: str,
        message: str,
        eliminated_sources: Optional[List[str]] = None,
        fallback_active_sources: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        eliminated_sources = eliminated_sources or []

        fit_source_indices = (
            list(self.current_fit_source_indices)
            if len(self.current_fit_source_indices) > 0
            else list(fallback_active_sources or [])
        )

        fit_species_indices = list(self.current_fit_species_indices)

        return {
            "row_index": self.current_row_index,
            "sample": self.current_sample,
            "status": "FAILED",
            "error_code": error_code,
            "message": message,
            "algorithm": self._algorithm_name(),
            "converged": False,

            "ambient_mass": float(self.amb_sp[1]) if self.amb_sp is not None else np.nan,
            "u_ambient_mass": float(self.u_amb_sp[1]) if self.u_amb_sp is not None else np.nan,

            "sum_sce": np.nan,
            "rsquare": np.nan,
            "chi_sqr": np.nan,
            "deg_free": 0,
            "pcnt_mass": np.nan,

            "eligible_dim": 0,
            "num_est_sources": 0,
            "frac_est": 0.0,

            "fit_species_indices": fit_species_indices,
            "fit_species_names": self._species_names_from_indices(fit_species_indices),
            "excluded_species_indices": list(self.current_excluded_species_indices),
            "excluded_species_names": self._species_names_from_indices(self.current_excluded_species_indices),

            "fit_source_indices": fit_source_indices,
            "source_names": self._source_names_from_indices(fit_source_indices),

            "eliminated_sources": list(eliminated_sources),

            "sc_e": None,
            "u_sc_e": None,
            "singular_values": None,
            "projections": None,

            "species_names": list(self.species_names),
            "species_measured": self.amb_sp[1:].copy() if self.amb_sp is not None else None,
            "species_unc": self.u_amb_sp[1:].copy() if self.u_amb_sp is not None else None,
            "species_calculated": None,
            "species_calc_unc": None,

            "m_pin": None,
            "m_pin_species_names": [],
            "m_pin_source_names": [],
        }


    def run_sample(
        self,
        sample_label: Optional[str] = None,
        row_pos: Optional[int] = None,
        fit_species: Optional[Sequence[Any]] = None,
        fit_sources: Optional[Sequence[Any]] = None,
        raise_on_error: bool = False
    ) -> Dict[str, Any]:
        if self.amb_df is None or self.amb_unc_df is None:
            err = CMBFitError(ERR_INPUT, "尚未加载输入数据，请先调用 load_data。")
            if raise_on_error:
                raise err
            return self._build_failure_result(err.code, err.message)

        if row_pos is None:
            if sample_label is None:
                err = CMBFitError(ERR_INPUT, "run_sample 必须提供 sample_label 或 row_pos。")
                if raise_on_error:
                    raise err
                return self._build_failure_result(err.code, err.message)

            matches = np.where(
                self.amb_df.iloc[:, 0].astype(str).to_numpy() == str(sample_label)
            )[0]

            if len(matches) == 0:
                err = CMBFitError(ERR_INPUT, f"未找到样品: {sample_label}")
                if raise_on_error:
                    raise err
                return self._build_failure_result(err.code, err.message)

            row_pos = int(matches[0])

        active_sources: List[int] = []
        eliminated_sources: List[str] = []

        try:
            self._reset_current_fit_state()

            self._prepare_sample_vectors(row_pos)

            self._log(f"正在处理样品第 {row_pos + 1} 行: {self.current_sample}")

            candidate_species = self._normalize_species_selection(fit_species)
            active_sources = self._normalize_source_selection(fit_sources)

            while True:
                self._set_fitting_elements_for_sample(
                    active_sources=active_sources,
                    candidate_species=candidate_species
                )

                self._perform_cmb_fit()

                if not self.config.auto_eliminate_negative:
                    break

                if self.n_pr_fit <= 1:
                    break

                worst_idx, worst_val = self._find_most_negative_source()

                if worst_idx is None:
                    break

                removed_source_global_idx = active_sources.pop(worst_idx)
                removed_source_name = self.source_names[removed_source_global_idx - 1]
                eliminated_sources.append(removed_source_name)

                self._log(
                    f"  自动剔除负贡献源: {removed_source_name} "
                    f"(SCE = {worst_val:.6g})"
                )

            self._calculate_estimable()

            result = self._build_success_result(eliminated_sources)
            self._log(f"  -> 成功: {self.current_sample}")
            return result

        except CMBFitError as e:
            self._log(f"  -> 失败: {self.current_sample} | {e.code} | {e.message}")

            if raise_on_error:
                raise

            return self._build_failure_result(
                error_code=e.code,
                message=e.message,
                eliminated_sources=eliminated_sources,
                fallback_active_sources=active_sources
            )

        except Exception as e:
            self._log(f"  -> 未预期错误: {self.current_sample} | {str(e)}")
            if self.config.verbose:
                traceback.print_exc()

            if raise_on_error:
                raise

            return self._build_failure_result(
                error_code=ERR_UNEXPECTED,
                message=str(e),
                eliminated_sources=eliminated_sources,
                fallback_active_sources=active_sources
            )

    def run_all(
        self,
        fit_species: Optional[Sequence[Any]] = None,
        fit_sources: Optional[Sequence[Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.amb_df is None:
            raise RuntimeError("尚未加载环境数据。")

        results: List[Dict[str, Any]] = []

        for row_pos in range(len(self.amb_df)):
            res = self.run_sample(
                row_pos=row_pos,
                fit_species=fit_species,
                fit_sources=fit_sources,
                raise_on_error=False
            )
            results.append(res)

        return results

    def export_to_excel(self, results_list: List[Dict[str, Any]], filename: str) -> None:
        if results_list is None:
            raise ValueError("results_list 不能为空。")

        out_dir = os.path.dirname(os.path.abspath(filename))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        param_rows = [{"参数": k, "值": v} for k, v in asdict(self.config).items()]
        param_rows.extend([
            {"参数": "n_species", "值": self.n_species},
            {"参数": "n_sources", "值": self.n_sources},
            {"参数": "n_samples", "值": len(self.amb_df) if self.amb_df is not None else 0},
            {"参数": "source_unc_zero_count_replaced", "值": self.source_unc_zero_count},
        ])
        df_params = pd.DataFrame(param_rows)

        summary_rows: List[Dict[str, Any]] = []
        error_rows: List[Dict[str, Any]] = []
        source_rows: List[Dict[str, Any]] = []
        species_rows: List[Dict[str, Any]] = []
        estimable_rows: List[Dict[str, Any]] = []
        singular_rows: List[Dict[str, Any]] = []
        mpin_rows: List[Dict[str, Any]] = []
        elimination_rows: List[Dict[str, Any]] = []

        for res in results_list:
            row_index = res.get("row_index", None)
            sample = res.get("sample", "")
            status = res.get("status", "")
            message = res.get("message", "")
            error_code = res.get("error_code", "")
            algorithm = res.get("algorithm", "")

            fit_species_names = res.get("fit_species_names", []) or []
            excluded_species_names = res.get("excluded_species_names", []) or []
            source_names = res.get("source_names", []) or []
            eliminated_sources = res.get("eliminated_sources", []) or []

            summary_rows.append({
                "环境行号(1-based)": (row_index + 1) if row_index is not None else np.nan,
                "样品": sample,
                "状态": status,
                "错误代码": error_code,
                "信息": message,
                "算法": algorithm,
                "收敛": res.get("converged", False),
                "环境总质量": res.get("ambient_mass", np.nan),
                "环境总质量不确定度": res.get("u_ambient_mass", np.nan),
                "计算总质量(sum_sce)": res.get("sum_sce", np.nan),
                "R_Square": res.get("rsquare", np.nan),
                "Chi_Square": res.get("chi_sqr", np.nan),
                "自由度": res.get("deg_free", np.nan),
                "百分质量": res.get("pcnt_mass", np.nan),
                "拟合物种数": len(fit_species_names),
                "拟合源数": len(source_names),
                "Eligible_Dim": res.get("eligible_dim", np.nan),
                "可估计源数": res.get("num_est_sources", np.nan),
                "Frac_Est": res.get("frac_est", np.nan),
                "当前拟合源": " | ".join(source_names),
                "自动剔除负源": " | ".join(eliminated_sources),
                "因MISS排除物种": " | ".join(excluded_species_names),
            })

            if status != "OK":
                error_rows.append({
                    "环境行号(1-based)": (row_index + 1) if row_index is not None else np.nan,
                    "样品": sample,
                    "错误代码": error_code,
                    "错误信息": message,
                    "算法": algorithm,
                    "已剔除负源": " | ".join(eliminated_sources),
                    "当前拟合源": " | ".join(source_names),
                    "因MISS排除物种": " | ".join(excluded_species_names),
                })
                continue
                
            sc_e = res["sc_e"]
            u_sc_e = res["u_sc_e"]
            fit_source_indices = res["fit_source_indices"]

            for j, src_name in enumerate(source_names):
                tstat = sc_e[j] / u_sc_e[j] if u_sc_e[j] > 0 else np.nan
                source_rows.append({
                    "环境行号(1-based)": row_index + 1,
                    "样品": sample,
                    "源全局索引(1-based)": fit_source_indices[j],
                    "源名称": src_name,
                    "SCE": sc_e[j],
                    "StdErr": u_sc_e[j],
                    "Tstat": tstat,
                    "算法": algorithm,
                })

            """
            # ---------- 物种浓度对比 ----------
            measured = res["species_measured"]
            measured_unc = res["species_unc"]
            calc = res["species_calculated"]
            calc_unc = res["species_calc_unc"]
            fit_species_indices = set(res["fit_species_indices"])

            for i, sp_name in enumerate(self.species_names, start=1):
                meas_i = measured[i - 1]
                umeas_i = measured_unc[i - 1]
                calc_i = calc[i - 1]
                ucalc_i = calc_unc[i - 1]

                residual = np.nan
                sig_res = np.nan
                res_over_sig = np.nan
                ratio_c_m = np.nan

                valid_meas = not self._is_miss_value(meas_i)
                valid_calc = not self._is_miss_value(calc_i)
                valid_umeas = not self._is_miss_value(umeas_i)
                valid_ucalc = not self._is_miss_value(ucalc_i)

                if valid_meas and valid_calc:
                    residual = calc_i - meas_i
                    if meas_i > 0:
                        ratio_c_m = calc_i / meas_i

                if valid_umeas and valid_ucalc:
                    sig_res = float(np.sqrt(ucalc_i ** 2 + umeas_i ** 2))
                    if valid_meas and valid_calc and sig_res > 0:
                        res_over_sig = residual / sig_res

                species_rows.append({
                    "环境行号(1-based)": row_index + 1,
                    "样品": sample,
                    "物种序号(1-based)": i,
                    "物种": sp_name,
                    "是否参与拟合": "是" if i in fit_species_indices else "否",
                    "实测": meas_i,
                    "实测不确定度": umeas_i,
                    "计算": calc_i,
                    "计算不确定度": ucalc_i,
                    "残差(计算-实测)": residual,
                    "残差综合不确定度": sig_res,
                    "残差/综合不确定度": res_over_sig,
                    "C/M比": ratio_c_m,
                })
            """

            projections = res["projections"]
            singular_values = res["singular_values"]

            for j, src_name in enumerate(source_names):
                estimable_rows.append({
                    "环境行号(1-based)": row_index + 1,
                    "样品": sample,
                    "源全局索引(1-based)": fit_source_indices[j],
                    "源名称": src_name,
                    "投影值": projections[j],
                    "对应奇异值": singular_values[j] if j < len(singular_values) else np.nan,
                    "是否可估计": "是" if projections[j] >= self.config.min_source_proj else "否",
                })

            for j, sval in enumerate(singular_values, start=1):
                inv_sval = (1.0 / sval) if sval != 0 else np.nan
                singular_rows.append({
                    "环境行号(1-based)": row_index + 1,
                    "样品": sample,
                    "序号": j,
                    "奇异值": sval,
                    "1/奇异值": inv_sval,
                })

            m_pin = res["m_pin"]
            mpin_species_names = res["m_pin_species_names"]
            mpin_source_names = res["m_pin_source_names"]

            if m_pin is not None:
                for i, sp_name in enumerate(mpin_species_names):
                    for j, src_name in enumerate(mpin_source_names):
                        mpin_rows.append({
                            "环境行号(1-based)": row_index + 1,
                            "样品": sample,
                            "物种": sp_name,
                            "源名称": src_name,
                            "mPIN": m_pin[i, j],
                        })

            for order, src_name in enumerate(eliminated_sources, start=1):
                elimination_rows.append({
                    "环境行号(1-based)": row_index + 1,
                    "样品": sample,
                    "剔除顺序": order,
                    "被剔除源": src_name,
                })

        df_summary = pd.DataFrame(summary_rows)
        df_errors = pd.DataFrame(error_rows)
        df_sources = pd.DataFrame(source_rows)
        # df_species = pd.DataFrame(species_rows)
        df_estimable = pd.DataFrame(estimable_rows)
        df_singular = pd.DataFrame(singular_rows)
        df_mpin = pd.DataFrame(mpin_rows)
        df_elimination = pd.DataFrame(elimination_rows)

        if df_errors.empty:
            df_errors = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "错误代码", "错误信息",
                "算法", "已剔除负源", "当前拟合源", "因MISS排除物种"
            ])

        if df_sources.empty:
            df_sources = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "源全局索引(1-based)",
                "源名称", "SCE", "StdErr", "Tstat", "算法"
            ])
        """
        if df_species.empty:
            df_species = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "物种序号(1-based)", "物种", "是否参与拟合",
                "实测", "实测不确定度", "计算", "计算不确定度",
                "残差(计算-实测)", "残差综合不确定度", "残差/综合不确定度", "C/M比"
            ])
        """
        if df_estimable.empty:
            df_estimable = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "源全局索引(1-based)",
                "源名称", "投影值", "对应奇异值", "是否可估计"
            ])

        if df_singular.empty:
            df_singular = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "序号", "奇异值", "1/奇异值"
            ])

        if df_mpin.empty:
            df_mpin = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "物种", "源名称", "mPIN"
            ])

        if df_elimination.empty:
            df_elimination = pd.DataFrame(columns=[
                "环境行号(1-based)", "样品", "剔除顺序", "被剔除源"
            ])

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_params.to_excel(writer, sheet_name="参数设置", index=False)
            df_summary.to_excel(writer, sheet_name="批量统计", index=False)
            df_errors.to_excel(writer, sheet_name="错误日志", index=False)
            df_sources.to_excel(writer, sheet_name="源贡献汇总", index=False)
            # df_species.to_excel(writer, sheet_name="物种浓度对比", index=False)
            df_estimable.to_excel(writer, sheet_name="可估计源", index=False)
            df_singular.to_excel(writer, sheet_name="奇异值", index=False)
            df_mpin.to_excel(writer, sheet_name="mPIN长表", index=False)
            df_elimination.to_excel(writer, sheet_name="负源剔除记录", index=False)

        self._log(f"结果已保存至: {filename}")



def run_single_source(
    source_file_path: str,
    receptor_file_path: str,
    output_file_path: str,
    config: CMBConfig,
    sheets: Dict[str, str],
) -> None:
    """处理单个源文件，输出结果到指定文件"""
    source_profile = read_table_auto(source_file_path, sheets["source_profile"])
    source_unc = read_table_auto(source_file_path, sheets["source_unc"])
    amb = read_table_auto(receptor_file_path, sheets["amb"])
    amb_unc = read_table_auto(receptor_file_path, sheets["amb_unc"])

    model = CMBModel(config)
    model.load_data(source_profile, source_unc, amb, amb_unc)
    results = model.run_all()
    model.export_to_excel(results, output_file_path)


def is_directory(path: str) -> bool:
    return os.path.isdir(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EPA‑CMB8.2 化学质量平衡受体模型（Python 版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个源文件
  python cmb_tool.py -s source.xlsx -r receptor.xlsx -o result.xlsx
  # 批量处理文件夹
  python cmb_tool.py -s ./sources -r receptor.xlsx -o ./results
        """
    )

    parser.add_argument(
        "-s", "--source",
        required=True,
        help="源文件路径（.xlsx/.csv）或包含多个源文件的文件夹路径。"
             "每个源文件必须包含源廓线 sheet 和源不确定度 sheet。"
    )
    parser.add_argument(
        "-r", "--receptor",
        required=True,
        help="受体（环境）文件路径（.xlsx/.csv），包含环境浓度和环境不确定度。"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出路径：单个文件时指定结果文件全名；批量模式时指定输出文件夹。"
    )

    parser.add_argument(
        "--sheet-source-profile",
        default="source_profile",
        help="源文件中源廓线所在的 sheet 名称（默认: source_profile）"
    )
    parser.add_argument(
        "--sheet-source-unc",
        default="source_unc",
        help="源文件中源不确定度所在的 sheet 名称（默认: source_unc）"
    )
    parser.add_argument(
        "--sheet-amb",
        default="amb",
        help="受体文件中环境浓度所在的 sheet 名称（默认: amb）"
    )
    parser.add_argument(
        "--sheet-amb-unc",
        default="amb_unc",
        help="受体文件中环境不确定度所在的 sheet 名称（默认: amb_unc）"
    )

    parser.add_argument(
        "--britt-luecke",
        action="store_true",
        default=False,
        help="启用 Britt‑Luecke 算法（默认关闭）"
    )
    parser.add_argument(
        "--no-auto-eliminate",
        action="store_true",
        default=False,
        help="关闭负源自动剔除功能（默认开启）"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="最大迭代次数（默认: 100）"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.01,
        help="收敛阈值（默认: 0.01）"
    )
    parser.add_argument(
        "--max-source-unc-pct",
        type=float,
        default=100.0,
        help="最大源不确定度百分比，用于可估计源判断（默认: 100.0）"
    )
    parser.add_argument(
        "--min-source-proj",
        type=float,
        default=0.1,
        help="最小投影阈值，用于可估计源判断（默认: 0.1）"
    )
    parser.add_argument(
        "--miss-value",
        type=float,
        default=-99.0,
        help="缺失值标记（默认: -99.0）"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-5,
        help="与 MISS 值比较时的容差（默认: 1e-5）"
    )
    parser.add_argument(
        "--source-unc-zero-replacement",
        type=float,
        default=1.0e-5,
        help="source_unc 中 0 的替换值（默认: 1e-5）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="打印详细运行日志"
    )

    args = parser.parse_args()

    config = CMBConfig(
        miss_value=args.miss_value,
        tol=args.tol,
        convergence_threshold=args.convergence_threshold,
        max_source_unc_pct=args.max_source_unc_pct,
        min_source_proj=args.min_source_proj,
        max_iterations=args.max_iter,
        use_britt_luecke=args.britt_luecke,
        auto_eliminate_negative=not args.no_auto_eliminate,
        source_unc_zero_replacement=args.source_unc_zero_replacement,
        verbose=args.verbose,
    )

    sheets = {
        "source_profile": args.sheet_source_profile,
        "source_unc": args.sheet_source_unc,
        "amb": args.sheet_amb,
        "amb_unc": args.sheet_amb_unc,
    }

    source_path = args.source
    receptor_path = args.receptor
    output_path = args.output

    if is_directory(source_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        elif not os.path.isdir(output_path):
            parser.error("批量模式时 output 必须为文件夹路径")

        extensions = (".xlsx", ".xls", ".xlsm", ".xlsb", ".csv")
        files = [
            f for f in os.listdir(source_path)
            if os.path.isfile(os.path.join(source_path, f)) and f.lower().endswith(extensions)
        ]
        if not files:
            print(f"警告：在 {source_path} 中未找到可处理的文件。", file=sys.stderr)
            sys.exit(1)

        for src_file in tqdm(files, desc="Processing sources"):
            full_source = os.path.join(source_path, src_file)
            out_name = os.path.splitext(src_file)[0] + ".xlsx"
            full_output = os.path.join(output_path, out_name)
            try:
                run_single_source(full_source, receptor_path, full_output, config, sheets)
            except Exception as e:
                print(f"处理 {src_file} 时出错: {e}", file=sys.stderr)
                continue

    else:
        if not os.path.isfile(source_path):
            parser.error(f"源文件不存在: {source_path}")
        if not os.path.isfile(receptor_path):
            parser.error(f"受体文件不存在: {receptor_path}")

        if os.path.isdir(output_path):
            out_name = os.path.splitext(os.path.basename(source_path))[0] + ".xlsx"
            full_output = os.path.join(output_path, out_name)
        else:
            full_output = output_path
            out_dir = os.path.dirname(full_output)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        run_single_source(source_path, receptor_path, full_output, config, sheets)
        print(f"结果已保存至: {full_output}")


if __name__ == "__main__":
    main()
