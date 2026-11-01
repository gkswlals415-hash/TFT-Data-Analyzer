# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import hashlib
import colorsys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


# ============================================================
# Page
# ============================================================
st.set_page_config(
    page_title="Transistor Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("Transistor Analyzer")
st.caption("Single / Batch TFT Transfer Characteristic Analysis")


# ============================================================
# Core analysis functions
# - migrated from the user's current Single / Batch analyzers
# ============================================================
def _excel_source(file_bytes: bytes):
    return io.BytesIO(file_bytes)



def display_centered_figure(fig):
    """Display a Matplotlib figure at about 80% of the page width."""
    _, center_col, _ = st.columns([1, 8, 1])
    with center_col:
        st.pyplot(fig, use_container_width=True)


def find_header_row(
    file_bytes: bytes,
    sheet_name: str,
    required_columns: list[str],
    max_scan_rows: int = 100,
) -> int:
    preview = pd.read_excel(
        _excel_source(file_bytes),
        sheet_name=sheet_name,
        header=None,
        nrows=max_scan_rows,
    )
    required = {str(c).strip() for c in required_columns}

    for i, row in preview.iterrows():
        values = {str(v).strip() for v in row.tolist() if pd.notna(v)}
        if required.issubset(values):
            return int(i)

    raise ValueError(
        f"첫 {max_scan_rows}행 안에서 필요한 열 {required_columns}을 찾지 못했습니다."
    )


def get_sheet_names(file_bytes: bytes) -> list[str]:
    xls = pd.ExcelFile(_excel_source(file_bytes))
    return list(xls.sheet_names)


def read_measurement_sheet(
    file_bytes: bytes,
    sheet_name: str,
    col_vg: str,
    col_id: str,
    col_ig: str,
    col_vd: str,
) -> tuple[pd.DataFrame, int]:
    header_row = find_header_row(file_bytes, sheet_name, [col_vg, col_id])

    df = pd.read_excel(
        _excel_source(file_bytes),
        sheet_name=sheet_name,
        header=header_row,
    )
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in [col_vg, col_id] if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열을 찾지 못했습니다: {missing}")

    keep = [
        c for c in [col_vg, col_id, col_ig, col_vd]
        if c and c in df.columns
    ]
    out = df[keep].copy()

    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=[col_vg, col_id]).reset_index(drop=True)

    if len(out) < 5:
        raise ValueError("유효한 Vg/Id 데이터 포인트가 너무 적습니다.")

    return out, header_row


def split_dual_sweep(vg: np.ndarray) -> list[tuple[str, slice]]:
    n = len(vg)
    if n < 6:
        return [("Sweep", slice(0, n))]

    vmin = np.nanmin(vg)
    vmax = np.nanmax(vg)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return [("Sweep", slice(0, n))]

    start = vg[0]
    if abs(start - vmin) <= abs(start - vmax):
        turn = int(np.nanargmax(vg))
    else:
        turn = int(np.nanargmin(vg))

    minimum_side = max(3, int(0.08 * n))
    if turn < minimum_side or turn > n - minimum_side:
        return [("Sweep", slice(0, n))]

    return [
        ("Forward", slice(0, turn + 1)),
        ("Reverse", slice(turn, n)),
    ]


def prepare_segment(
    vg,
    raw_id,
    raw_ig=None,
    raw_vd=None,
    use_abs_id=True,
) -> pd.DataFrame:
    seg = pd.DataFrame({
        "Vg": pd.to_numeric(pd.Series(vg), errors="coerce"),
        "Id_raw": pd.to_numeric(pd.Series(raw_id), errors="coerce"),
    })

    if raw_ig is not None:
        seg["Ig_raw"] = pd.to_numeric(pd.Series(raw_ig), errors="coerce")
    if raw_vd is not None:
        seg["Vd_raw"] = pd.to_numeric(pd.Series(raw_vd), errors="coerce")

    seg = seg.dropna(subset=["Vg", "Id_raw"]).reset_index(drop=True)
    if len(seg) < 3:
        raise ValueError("Sweep 구간의 유효 데이터가 너무 적습니다.")

    seg["Id_analysis"] = seg["Id_raw"].abs() if use_abs_id else seg["Id_raw"]

    agg = {
        "Id_raw": "mean",
        "Id_analysis": "mean",
    }
    if "Ig_raw" in seg.columns:
        agg["Ig_raw"] = "mean"
    if "Vd_raw" in seg.columns:
        agg["Vd_raw"] = "mean"

    seg = seg.groupby("Vg", sort=False, as_index=False).agg(agg)

    if len(seg) < 3:
        raise ValueError("중복 Vg 제거 후 데이터가 너무 적습니다.")

    return seg


def analyze_sweep(
    seg: pd.DataFrame,
    smoothing_window: int = 5,
    ss_window: int = 7,
    constant_current_A: float = 1e-6,
) -> tuple[dict, pd.DataFrame]:
    data = seg.copy()

    # Smoothing
    win = max(1, int(smoothing_window))
    if win % 2 == 0:
        win += 1
    max_win = len(data)
    if max_win % 2 == 0:
        max_win -= 1
    win = min(win, max(1, max_win))

    if win >= 3:
        data["Id_smooth"] = (
            data["Id_analysis"]
            .rolling(window=win, center=True, min_periods=1)
            .mean()
        )
    else:
        data["Id_smooth"] = data["Id_analysis"]

    vg = data["Vg"].to_numpy(dtype=float)
    ids = data["Id_smooth"].to_numpy(dtype=float)

    # Vg sweep step: median of non-zero adjacent voltage differences
    vg_diffs = np.abs(np.diff(vg))
    vg_diffs = vg_diffs[np.isfinite(vg_diffs) & (vg_diffs > 0)]
    vg_step = float(np.nanmedian(vg_diffs)) if len(vg_diffs) else np.nan

    # gm
    gm = np.gradient(ids, vg)
    data["gm"] = gm

    finite = np.isfinite(gm) & np.isfinite(ids) & np.isfinite(vg)
    valid_indices = np.where(finite)[0]
    if len(valid_indices) == 0:
        raise ValueError("gm을 계산할 수 없습니다.")

    if len(valid_indices) >= 7:
        candidate = valid_indices[1:-1]
    else:
        candidate = valid_indices
    if len(candidate) == 0:
        candidate = valid_indices

    idx = int(candidate[np.nanargmax(gm[candidate])])
    gm_max = float(gm[idx])
    vg_gm = float(vg[idx])
    id_gm = float(ids[idx])

    if not np.isfinite(gm_max) or abs(gm_max) < 1e-30:
        raise ValueError("gm_max가 0에 가까워 Vth를 계산할 수 없습니다.")

    # gm-max tangent Vth
    vth = vg_gm - (id_gm / gm_max)

    # SS
    id_for_ss = np.abs(data["Id_smooth"].to_numpy(dtype=float))
    valid_ss = np.isfinite(vg) & np.isfinite(id_for_ss) & (id_for_ss > 0)
    vg_ss = vg[valid_ss]
    id_ss = id_for_ss[valid_ss]

    ss_window = max(3, int(ss_window))
    if ss_window % 2 == 0:
        ss_window += 1

    best_slope = np.nan
    best_ss = np.nan
    best_vg_center = np.nan

    if len(vg_ss) >= ss_window:
        log_id = np.log10(id_ss)
        for i in range(len(vg_ss) - ss_window + 1):
            x = vg_ss[i:i + ss_window]
            y = log_id[i:i + ss_window]
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                continue
            slope, _intercept = np.polyfit(x, y, 1)
            if slope > 0:
                if np.isnan(best_slope) or slope > best_slope:
                    best_slope = float(slope)
                    best_ss = float(1.0 / slope)
                    best_vg_center = float(np.mean(x))

    ss_v_dec = best_ss
    ss_mv_dec = best_ss * 1000 if np.isfinite(best_ss) else np.nan

    # Constant-current Vth
    # |Id| = Iref가 되는 Vg를 log10(|Id|)-Vg 공간에서 선형 보간합니다.
    # 노이즈로 교차점이 여러 개면 SS maximum-slope 중심에 가장 가까운 교차점을 선택합니다.
    iref = float(constant_current_A)
    if not np.isfinite(iref) or iref <= 0:
        raise ValueError("Constant current Iref는 0보다 커야 합니다.")

    id_cc = np.abs(data["Id_analysis"].to_numpy(dtype=float))
    valid_cc = np.isfinite(vg) & np.isfinite(id_cc) & (id_cc > 0)
    vg_cc = vg[valid_cc]
    id_cc = id_cc[valid_cc]
    cc_candidates = []

    if len(vg_cc) >= 2:
        log_cc = np.log10(id_cc)
        target = np.log10(iref)
        for i in range(len(vg_cc) - 1):
            x1, x2 = float(vg_cc[i]), float(vg_cc[i + 1])
            y1, y2 = float(log_cc[i]), float(log_cc[i + 1])
            d1, d2 = y1 - target, y2 - target

            if d1 == 0:
                cc_candidates.append(x1)
            if d2 == 0:
                cc_candidates.append(x2)

            if d1 * d2 < 0:
                if y2 != y1:
                    x_cross = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
                else:
                    x_cross = (x1 + x2) / 2.0
                cc_candidates.append(float(x_cross))

    if cc_candidates:
        cc_candidates = np.asarray(cc_candidates, dtype=float)
        if np.isfinite(best_vg_center):
            vth_cc = float(cc_candidates[np.argmin(np.abs(cc_candidates - best_vg_center))])
        else:
            vth_cc = float(cc_candidates[0])
    else:
        vth_cc = np.nan

    # ON/OFF
    id_positive = np.abs(data["Id_analysis"].to_numpy(dtype=float))
    finite_positive = id_positive[
        np.isfinite(id_positive) & (id_positive > 0)
    ]
    if len(finite_positive):
        ion = float(np.nanmax(finite_positive))
        ioff = float(np.nanmin(finite_positive))
        onoff = ion / ioff if ioff > 0 else np.nan
    else:
        ion = ioff = onoff = np.nan

    result = {
        "Vth_V": vth,
        "gm_max_S": gm_max,
        "Vg_at_gmmax_V": vg_gm,
        "Id_at_gmmax_A": id_gm,
        "SS_V_dec": ss_v_dec,
        "SS_mV_dec": ss_mv_dec,
        "SS_Vg_center_V": best_vg_center,
        "Vth_CC_V": vth_cc,
        "Constant_Current_A": iref,
        "Ion_A": ion,
        "Ioff_A": ioff,
        "On_Off": onoff,
        "Vg_Step_V": vg_step,
        "Smoothing_Window": win,
        "SS_Window": ss_window,
        "gm_index": idx,
    }
    return result, data


def analyze_reverse_for_vth(
    seg: pd.DataFrame,
    smoothing_window: int = 5,
    constant_current_A: float = 1e-6,
) -> tuple[dict, pd.DataFrame]:
    """Reverse: Id를 스무딩해 Vth 매칭만 준비. gm/SS/전류비/이동도는 계산하지 않습니다."""
    data = seg.copy()
    win = max(1, int(smoothing_window))
    if win % 2 == 0:
        win += 1
    max_win = len(data) - (len(data) % 2 == 0)
    win = min(win, max(1, max_win))
    if win >= 3:
        data["Id_smooth"] = (
            data["Id_analysis"].rolling(window=win, center=True, min_periods=1).mean()
        )
    else:
        data["Id_smooth"] = data["Id_analysis"]
    vg = data["Vg"].to_numpy(dtype=float)
    steps = np.abs(np.diff(vg))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    iref = float(constant_current_A)
    if not np.isfinite(iref) or iref <= 0:
        raise ValueError("Constant current Iref는 0보다 커야 합니다.")
    return {
        "Vth_V": np.nan,
        "Vth_CC_V": np.nan,
        "Constant_Current_A": iref,
        "Vg_Step_V": float(np.median(steps)) if len(steps) else np.nan,
        "Smoothing_Window": win,
    }, data


def interpolate_current_at_vg(
    analyzed: pd.DataFrame,
    target_vg: float,
    current_column: str = "Id_smooth",
) -> float:
    """특정 Vg에서의 |Id|를 log10(|Id|)-Vg 공간에서 보간합니다."""
    if not np.isfinite(target_vg) or current_column not in analyzed.columns:
        return np.nan

    vg = analyzed["Vg"].to_numpy(dtype=float)
    current = np.abs(analyzed[current_column].to_numpy(dtype=float))
    valid = np.isfinite(vg) & np.isfinite(current) & (current > 0)
    vg = vg[valid]
    current = current[valid]

    if len(vg) < 2:
        return np.nan

    order = np.argsort(vg)
    vg = vg[order]
    current = current[order]

    if target_vg < np.nanmin(vg) or target_vg > np.nanmax(vg):
        return np.nan

    log_i = np.log10(current)
    log_target_i = np.interp(float(target_vg), vg, log_i)
    return float(10 ** log_target_i)


def find_vg_at_current(
    analyzed: pd.DataFrame,
    target_current: float,
    preferred_vg: float | None = None,
    current_column: str = "Id_smooth",
) -> float:
    """주어진 |Id|와 같은 전류가 흐르는 Vg를 log-current 공간에서 찾습니다."""
    if (
        not np.isfinite(target_current)
        or target_current <= 0
        or current_column not in analyzed.columns
    ):
        return np.nan

    vg = analyzed["Vg"].to_numpy(dtype=float)
    current = np.abs(analyzed[current_column].to_numpy(dtype=float))
    valid = np.isfinite(vg) & np.isfinite(current) & (current > 0)
    vg = vg[valid]
    current = current[valid]

    if len(vg) < 2:
        return np.nan

    log_i = np.log10(current)
    target = np.log10(float(target_current))
    candidates = []

    for i in range(len(vg) - 1):
        x1, x2 = float(vg[i]), float(vg[i + 1])
        y1, y2 = float(log_i[i]), float(log_i[i + 1])
        d1, d2 = y1 - target, y2 - target

        if np.isclose(d1, 0.0, atol=1e-12):
            candidates.append(x1)
        if np.isclose(d2, 0.0, atol=1e-12):
            candidates.append(x2)

        if d1 * d2 < 0:
            if not np.isclose(y2, y1):
                x_cross = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_cross = (x1 + x2) / 2.0
            candidates.append(float(x_cross))

    if not candidates:
        return np.nan

    cand = np.asarray(candidates, dtype=float)
    cand = cand[np.isfinite(cand)]
    if len(cand) == 0:
        return np.nan

    if preferred_vg is not None and np.isfinite(preferred_vg):
        return float(cand[np.argmin(np.abs(cand - float(preferred_vg)))])

    return float(cand[0])


def add_mobility(
    result: dict,
    analyzed: pd.DataFrame,
    W_um: float,
    L_um: float,
    eps_r: float,
    tox_nm: float,
    eps0: float,
) -> dict:
    if W_um <= 0 or L_um <= 0:
        raise ValueError("W와 L은 0보다 커야 합니다.")
    if eps_r <= 0 or tox_nm <= 0 or eps0 <= 0:
        raise ValueError("eps_r, tox, eps0는 0보다 커야 합니다.")

    tox_cm = tox_nm * 1e-7
    Cox = eps0 * eps_r / tox_cm

    if "Vd_raw" in analyzed.columns:
        Vd = float(np.nanmedian(np.abs(analyzed["Vd_raw"].to_numpy(dtype=float))))
    else:
        Vd = np.nan

    if np.isfinite(Vd) and Vd > 0:
        mu_fe = result["gm_max_S"] * (L_um / W_um) / (Cox * Vd)
    else:
        mu_fe = np.nan

    result = dict(result)
    result["Cox_F_cm2"] = Cox
    result["mu_FE_cm2_Vs"] = mu_fe
    result["Vd_used_V"] = Vd
    return result


def analyze_single_file(
    file_bytes: bytes,
    sheet_name: str,
    *,
    col_vg: str,
    col_id: str,
    col_ig: str,
    col_vd: str,
    smoothing_window: int,
    ss_window: int,
    constant_current_A: float,
    use_abs_id: bool,
    W_um: float,
    L_um: float,
    eps_r: float,
    tox_nm: float,
    eps0: float,
):
    df, header_row = read_measurement_sheet(
        file_bytes,
        sheet_name,
        col_vg,
        col_id,
        col_ig,
        col_vd,
    )

    vg = df[col_vg].to_numpy(dtype=float)
    id_raw = df[col_id].to_numpy(dtype=float)
    ig_raw = df[col_ig].to_numpy(dtype=float) if col_ig in df.columns else None
    vd_raw = df[col_vd].to_numpy(dtype=float) if col_vd in df.columns else None

    sweeps = split_dual_sweep(vg)
    results = {}
    frames = {}

    for label, sl in sweeps:
        seg = prepare_segment(
            vg[sl],
            id_raw[sl],
            ig_raw[sl] if ig_raw is not None else None,
            vd_raw[sl] if vd_raw is not None else None,
            use_abs_id=use_abs_id,
        )
        if label == "Reverse":
            result, analyzed = analyze_reverse_for_vth(
                seg,
                smoothing_window=smoothing_window,
                constant_current_A=constant_current_A,
            )
        else:
            result, analyzed = analyze_sweep(
                seg,
                smoothing_window=smoothing_window,
                ss_window=ss_window,
                constant_current_A=constant_current_A,
            )
            result = add_mobility(
                result, analyzed,
                W_um=W_um, L_um=L_um,
                eps_r=eps_r, tox_nm=tox_nm, eps0=eps0,
            )

        results[label] = result
        frames[label] = analyzed

    # Primary Vth / Hysteresis
    # Forward: 기존 gm-max tangent Vth
    # Reverse: Forward Vth 위치에서의 |Id|와 동일한 전류가 흐르는 Vg
    if "Forward" in results and "Reverse" in results:
        f = results["Forward"]
        r = results["Reverse"]

        match_current = interpolate_current_at_vg(
            frames["Forward"],
            f["Vth_V"],
            current_column="Id_smooth",
        )
        matched_reverse_vth = find_vg_at_current(
            frames["Reverse"],
            match_current,
            preferred_vg=f["Vth_V"],
            current_column="Id_smooth",
        )

        f["Vth_Primary_Method"] = "Forward gm-max tangent"
        r["Vth_Primary_Method"] = "Reverse same-|Id| as Forward Vth"
        f["Vth_Match_Current_A"] = match_current
        r["Vth_Match_Current_A"] = match_current
        r["Vth_V"] = matched_reverse_vth
        # Reverse CC Vth는 gm/SS 계산 없이 원본 |Id|와 Iref의 교차점으로만 산출합니다.
        # 복수 교차점이면 Forward CC Vth에 가장 가까운 지점을 선택합니다.
        r["Vth_CC_V"] = find_vg_at_current(
            frames["Reverse"], float(constant_current_A),
            preferred_vg=f["Vth_CC_V"], current_column="Id_analysis",
        )

    return results, frames, df, header_row


def find_default_measurement_sheet(
    file_bytes: bytes,
    col_vg: str,
    col_id: str,
) -> str:
    sheets = get_sheet_names(file_bytes)
    if not sheets:
        raise ValueError("Sheet가 없습니다.")

    for sheet in reversed(sheets):
        try:
            find_header_row(file_bytes, sheet, [col_vg, col_id], max_scan_rows=100)
            return sheet
        except Exception:
            continue

    raise ValueError(
        f"측정 열({col_vg}, {col_id})이 있는 Sheet를 찾지 못했습니다."
    )


def analyze_batch_file(
    file_bytes: bytes,
    file_name: str,
    sheet_name: str,
    *,
    device_label: str | None = None,
    col_vg: str,
    col_id: str,
    col_ig: str,
    col_vd: str,
    smoothing_window: int,
    ss_window: int,
    constant_current_A: float,
    use_abs_id: bool,
    W_um: float,
    L_um: float,
    eps_r: float,
    tox_nm: float,
    eps0: float,
):
    results, frames, df, header_row = analyze_single_file(
        file_bytes,
        sheet_name,
        col_vg=col_vg,
        col_id=col_id,
        col_ig=col_ig,
        col_vd=col_vd,
        smoothing_window=smoothing_window,
        ss_window=ss_window,
        constant_current_A=constant_current_A,
        use_abs_id=use_abs_id,
        W_um=W_um,
        L_um=L_um,
        eps_r=eps_r,
        tox_nm=tox_nm,
        eps0=eps0,
    )

    if "Forward" not in results or "Reverse" not in results:
        raise ValueError("Forward/Reverse 결과를 모두 얻지 못했습니다.")

    f = results["Forward"]
    r = results["Reverse"]
    hysteresis = abs(r["Vth_V"] - f["Vth_V"])
    # CC 관련 결과는 소자별 결과표에 넣지 않습니다. (내부 계산은 기존 유지)

    path = Path(file_name)
    row = {
        # 내부에서는 파일/시트 및 매칭 전류를 유지해 그래프와 평균 계산에 활용합니다.
        "Device": device_label if device_label else path.stem,
        "File": path.name,
        "Sheet": sheet_name,
        "Vth F (V)": f["Vth_V"],
        "Vth R (V)": r["Vth_V"],
        "Vth Hys (V)": hysteresis,
        "Vth match current (A)": f.get("Vth_Match_Current_A", np.nan),
        "gm_max (S)": f["gm_max_S"],
        "SS (mV/dec)": f["SS_mV_dec"],
        "muFE (cm2/Vs)": f["mu_FE_cm2_Vs"],
        "ON/OFF": f["On_Off"],
        "VDS (V)": f["Vd_used_V"],
        "Vg@gmmax F (V)": f["Vg_at_gmmax_V"],
        "Id@gmmax F (A)": f["Id_at_gmmax_A"],
        "Cox (F/cm2)": f["Cox_F_cm2"],
        "Header Row": header_row + 1,
        "Data Points": len(df),
    }

    return row, frames


# ============================================================
# Formatting / plotting / export helpers
# ============================================================
def fmt_num(value, decimals=3):
    try:
        if np.isfinite(value):
            return f"{value:.{decimals}f}"
    except Exception:
        pass
    return "N/A"


def fmt_sci(value):
    try:
        if np.isfinite(value):
            return f"{value:.6e}"
    except Exception:
        pass
    return "N/A"


def gate_current_color(drain_color):
    """소자별 Id 색상에 대응하는, 확실히 구분되는 Ig 색상을 반환합니다."""
    hue, saturation, value = colorsys.rgb_to_hsv(*to_rgb(drain_color))
    return colorsys.hsv_to_rgb(
        (hue + 0.5) % 1.0, max(0.60, saturation), max(0.65, value)
    )


def make_batch_figures(curves: dict, figsize: tuple[float, float] = (8, 6)):
    # Linear
    fig1 = Figure(figsize=figsize, dpi=100)
    ax1 = fig1.add_subplot(111)
    for item in curves.values():
        device = item["Device"]
        f = item["Forward"]
        r = item["Reverse"]
        line_f, = ax1.plot(f["Vg"], f["Id_analysis"], label=device, linewidth=1.5)
        ax1.plot(
            r["Vg"], r["Id_analysis"], linestyle="--", linewidth=1.2,
            color=line_f.get_color(), label="_nolegend_"
        )
    ax1.set_xlabel("Gate Voltage, Vg (V)")
    ax1.set_ylabel("|Drain Current|, |Id| (A)")
    ax1.set_title("All Devices - Linear Id-Vg (solid F / dashed R)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(title="Device", fontsize=8, ncol=max(1, min(5, len(curves))))
    fig1.tight_layout()

    # Log Id + Ig
    fig2 = Figure(figsize=figsize, dpi=100)
    ax2 = fig2.add_subplot(111)
    device_handles = []
    any_ig = False
    any_vth_match = False

    for item in curves.values():
        device = item["Device"]
        f = item["Forward"]
        r = item["Reverse"]

        id_f = np.abs(f["Id_raw"].to_numpy(dtype=float))
        id_r = np.abs(r["Id_raw"].to_numpy(dtype=float))
        id_f[id_f <= 0] = np.nan
        id_r[id_r <= 0] = np.nan

        line_id_f, = ax2.semilogy(f["Vg"], id_f, linestyle="-", linewidth=1.7, label=device)
        color = line_id_f.get_color()
        ax2.semilogy(r["Vg"], id_r, linestyle="--", linewidth=1.5, color=color, label="_nolegend_")

        if "Ig_raw" in f.columns and "Ig_raw" in r.columns:
            ig_f = np.abs(f["Ig_raw"].to_numpy(dtype=float))
            ig_r = np.abs(r["Ig_raw"].to_numpy(dtype=float))
            ig_f[ig_f <= 0] = np.nan
            ig_r[ig_r <= 0] = np.nan
            ig_color = gate_current_color(color)
            # Drain / Gate는 서로 다른 색, Forward는 실선, Reverse는 점선.
            ax2.semilogy(f["Vg"], ig_f, linestyle="-", linewidth=1.7,
                         color=ig_color, alpha=0.95, label="_nolegend_")
            ax2.semilogy(r["Vg"], ig_r, linestyle="--", linewidth=1.5,
                         color=ig_color, alpha=0.95, label="_nolegend_")
            any_ig = True

        # Primary Vth 두 점은 *동일한 |Id|*에서 구했습니다.
        # F: gm-max tangent Vth, R: F Vth의 스무딩 전류와 일치하는 Vg.
        # 곡선은 기존 Batch 방식(원본 |Id|)을 유지하지만, 두 표시점은
        # 분석에 쓰인 스무딩 |Id| 기반 값이므로 곡선과 미세한 차이가 날 수 있습니다.
        vth_f = float(item.get("Vth_F", np.nan))
        vth_r = float(item.get("Vth_R", np.nan))
        match_i = float(item.get("Vth_match_current", np.nan))
        if np.isfinite(vth_f) and np.isfinite(vth_r) and np.isfinite(match_i) and match_i > 0:
            any_vth_match = True
            ax2.plot(
                [vth_f, vth_r], [match_i, match_i],
                color=color, linestyle=(0, (2, 3)), linewidth=1.7,
                alpha=0.95, zorder=6, label="_nolegend_",
            )
            ax2.scatter(
                [vth_f], [match_i], marker="o", s=85,
                facecolors="white", edgecolors=color, linewidths=2,
                zorder=8, label="_nolegend_",
            )
            ax2.scatter(
                [vth_r], [match_i], marker="D", s=67,
                facecolors=color, edgecolors="black", linewidths=0.8,
                zorder=9, label="_nolegend_",
            )

        device_handles.append(Line2D([0], [0], color=color, linewidth=1.8, linestyle="-", label=device))

    ax2.set_xlabel("Gate Voltage, Vg (V)")
    ax2.set_ylabel("Current, |Id| / |Ig| (A)")
    ax2.set_title("All Devices - Log Id + Ig - Vg")
    ax2.grid(True, which="both", alpha=0.25)

    if device_handles:
        device_legend = ax2.legend(
            handles=device_handles,
            title="Device",
            fontsize=8,
            ncol=max(1, min(5, len(device_handles))),
            loc="upper left",
        )
        ax2.add_artist(device_legend)

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.7, label="Id Forward"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Id Reverse"),
    ]
    if any_ig:
        style_handles.extend([
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.7, label="Ig Forward"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Ig Reverse"),
        ])
    if any_vth_match:
        style_handles.extend([
            Line2D([0], [0], color="black", marker="o", markersize=7,
                   markerfacecolor="white", linestyle="None", label="Vth F (gm-max)"),
            Line2D([0], [0], color="black", marker="D", markersize=6,
                   linestyle="None", label="Vth R (same-current)"),
            Line2D([0], [0], color="black", linestyle=(0, (2, 3)),
                   linewidth=1.5, label="Same-current Vth link"),
        ])
    ax2.legend(handles=style_handles, title="Current / Sweep / Vth", fontsize=8, loc="lower right")
    fig2.tight_layout()

    # gm - Forward only
    fig3 = Figure(figsize=figsize, dpi=100)
    ax3 = fig3.add_subplot(111)
    gm_f_handles = []

    for item in curves.values():
        device = item["Device"]
        f = item["Forward"]

        line_f, = ax3.plot(
            f["Vg"], f["gm"], linewidth=1.5, label=device
        )
        device_color = line_f.get_color()

        x_f = item.get("Vg_gmmax_F", np.nan)
        y_f = item.get("gmmax_F", np.nan)
        if np.isfinite(x_f) and np.isfinite(y_f):
            ax3.scatter(
                [x_f], [y_f],
                s=52, marker="o",
                color=device_color,
                edgecolors="black", linewidths=0.6,
                zorder=6, label="_nolegend_",
            )

        gm_f_handles.append(
            Line2D([0], [0], color=device_color, linewidth=1.8, label=device)
        )

    ax3.set_xlabel("Gate Voltage, Vg (V)")
    ax3.set_ylabel("Transconductance, gm (S)")
    ax3.set_title("All Devices - Forward gm-Vg")
    ax3.grid(True, alpha=0.25)

    if gm_f_handles:
        dev_leg_f = ax3.legend(
            handles=gm_f_handles,
            title="Device",
            fontsize=8,
            ncol=max(1, min(5, len(gm_f_handles))),
            loc="upper left",
        )
        ax3.add_artist(dev_leg_f)

    ax3.legend(
        handles=[
            Line2D(
                [0], [0], color="black", linestyle="-", linewidth=1.5,
                marker="o", markersize=6, markeredgecolor="black",
                label="gm_max",
            )
        ],
        fontsize=8,
        loc="lower right",
    )
    fig3.tight_layout()

    return fig1, fig2, fig3



def figure_to_png_bytes(fig: Figure, dpi: int = 300) -> bytes:
    """Matplotlib Figure를 고해상도 PNG bytes로 변환합니다."""
    output = io.BytesIO()
    # 여백을 잘라 비율이 바뀌지 않도록 저장 시 원본 캔버스를 유지합니다.
    fig.savefig(output, format="png", dpi=dpi)
    output.seek(0)
    return output.getvalue()


def build_figures_zip(figures, file_names: list[str], dpi: int = 300) -> bytes:
    """여러 Figure를 PNG로 변환하여 하나의 ZIP 파일로 묶습니다."""
    if len(figures) != len(file_names):
        raise ValueError("그래프 개수와 파일 이름 개수가 일치하지 않습니다.")

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fig, file_name in zip(figures, file_names):
            png_bytes = figure_to_png_bytes(fig, dpi=dpi)
            zf.writestr(file_name, png_bytes)

    output.seek(0)
    return output.getvalue()


def statistics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Batch 특성값 통계; 표시하지 않는 매칭 전류도 평균과 분산에 사용합니다."""
    metrics = [
        "Vth F (V)", "Vth R (V)", "Vth Hys (V)",
        "Vth match current (A)", "VDS (V)",
        "gm_max (S)", "SS (mV/dec)", "muFE (cm2/Vs)", "ON/OFF",
    ]

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue

        values = pd.to_numeric(df[metric], errors="coerce")
        values = values[np.isfinite(values)]

        if len(values):
            mean = float(values.mean())
            minimum = float(values.min())
            maximum = float(values.max())
        else:
            mean = minimum = maximum = np.nan

        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        variance = float(values.var(ddof=1)) if len(values) > 1 else np.nan
        cv = (std / abs(mean) * 100.0) if (np.isfinite(std) and np.isfinite(mean) and abs(mean) > 1e-30) else np.nan
        value_range = maximum - minimum if np.isfinite(maximum) and np.isfinite(minimum) else np.nan

        rows.append({
            "Metric": metric,
            "N": int(values.count()),
            "Mean": mean,
            "Std": std,
            "Variance": variance,
            "CV (%)": cv,
            "Min": minimum,
            "Max": maximum,
            "Range": value_range,
        })
    return pd.DataFrame(rows)


VARIABILITY_METRICS = {
    "Vth (F gm-max / R same-current)": {
        "columns": ["Vth F (V)", "Vth R (V)"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "Vth (V)",
        "log": False,
    },
    "SS": {
        "columns": ["SS (mV/dec)"],
        "labels": ["Forward"],
        "ylabel": "SS (mV/dec)",
        "log": False,
    },
    "Mobility μFE": {
        "columns": ["muFE (cm2/Vs)"],
        "labels": ["Forward"],
        "ylabel": "μFE (cm²/V·s)",
        "log": False,
    },
    "gm_max": {
        "columns": ["gm_max (S)"],
        "labels": ["Forward"],
        "ylabel": "gm_max (S)",
        "log": False,
    },
    "ON/OFF ratio": {
        "columns": ["ON/OFF"],
        "labels": ["Forward"],
        "ylabel": "ON/OFF ratio",
        "log": True,
    },
    "Vth Hysteresis (F gm-max / R same-current)": {
        "columns": ["Vth Hys (V)"],
        "labels": ["Hysteresis"],
        "ylabel": "Vth Hysteresis (V)",
        "log": False,
    },
    "Vth match current": {
        "columns": ["Vth match current (A)"],
        "labels": ["Match current"],
        "ylabel": "Match current (A)",
        "log": True,
    },
    "VDS": {
        "columns": ["VDS (V)"],
        "labels": ["VDS"],
        "ylabel": "VDS (V)",
        "log": False,
    },
}

def make_variability_figure(
    df: pd.DataFrame, metric_name: str, figsize: tuple[float, float] = (8, 6)
) -> Figure:
    """소자별 특성값과 평균 ± 1σ 범위를 함께 표시합니다."""
    info = VARIABILITY_METRICS[metric_name]
    columns = info["columns"]
    labels = info["labels"]

    fig = Figure(figsize=figsize, dpi=110)
    ax = fig.add_subplot(111)

    x = np.arange(len(df), dtype=float)
    device_names = df["Device"].astype(str).tolist()

    for col, label in zip(columns, labels):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        line, = ax.plot(
            x, y,
            marker="o", markersize=5, linewidth=1.2,
            label=label,
        )

        finite = y[np.isfinite(y)]
        if len(finite):
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            ax.axhline(
                mean,
                color=line.get_color(),
                linestyle=":", linewidth=1.25, alpha=0.9,
            )

            # 평균 ± 1σ 구간을 옅은 band로 표시
            if np.isfinite(std):
                low = mean - std
                high = mean + std
                if (not info["log"]) or low > 0:
                    ax.fill_between(
                        [-0.4, max(len(df) - 0.6, 0.6)],
                        [low, low], [high, high],
                        color=line.get_color(), alpha=0.08,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(device_names, rotation=45, ha="right")
    ax.set_xlabel("Device")
    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"Device-to-device variation - {metric_name}")
    ax.grid(True, alpha=0.25)
    if info["log"]:
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.legend(title="Sweep")
    fig.tight_layout()
    return fig


def format_statistics_for_display(stats_df: pd.DataFrame) -> pd.DataFrame:
    """통계표를 읽기 쉬운 표기법으로 바꾼 화면 표시용 DataFrame."""
    out = stats_df.copy()
    numeric_cols = ["Mean", "Std", "Variance", "Min", "Max", "Range"]

    def _auto_fmt(v):
        try:
            if not np.isfinite(v):
                return "N/A"
            av = abs(float(v))
            if av != 0 and (av < 1e-3 or av >= 1e5):
                return f"{v:.6e}"
            return f"{v:.6g}"
        except Exception:
            return "N/A"

    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].apply(_auto_fmt)

    if "CV (%)" in out.columns:
        out["CV (%)"] = out["CV (%)"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) and np.isfinite(v) else "N/A"
        )

    return out


OUTLIER_METRICS = {
    "Vth F (gm-max)": "Vth F (V)",
    "Vth R (same-current)": "Vth R (V)",
    "Vth Hysteresis (F gm-max / R same-current)": "Vth Hys (V)",
    "gm_max": "gm_max (S)",
    "SS": "SS (mV/dec)",
    "μFE": "muFE (cm2/Vs)",
    "ON/OFF": "ON/OFF",
    "Vth match current": "Vth match current (A)",
    "VDS": "VDS (V)",
}

def detect_iqr_outliers(
    df: pd.DataFrame,
    columns: list[str],
    multiplier: float = 1.5,
) -> tuple[set[str], pd.DataFrame]:
    """선택한 특성값에서 IQR 기준 이상치를 찾습니다.

    한 소자가 선택한 특성 중 하나라도 이상치이면 해당 소자를 제외 대상으로 표시합니다.
    원본 데이터 자체는 삭제하지 않습니다.
    """
    flagged: set[str] = set()
    details: list[dict] = []

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]

        # 너무 적은 표본에서는 IQR 판정이 의미가 약하므로 자동 제거하지 않음
        if len(finite) < 4:
            continue

        q1 = float(finite.quantile(0.25))
        q3 = float(finite.quantile(0.75))
        iqr = q3 - q1

        if not np.isfinite(iqr) or iqr <= 0:
            continue

        lower = q1 - float(multiplier) * iqr
        upper = q3 + float(multiplier) * iqr

        for idx, value in values.items():
            if not np.isfinite(value):
                continue
            if value < lower or value > upper:
                device = str(df.loc[idx, "Device"])
                flagged.add(device)
                details.append({
                    "Device": device,
                    "Metric": col,
                    "Value": float(value),
                    "Lower bound": lower,
                    "Upper bound": upper,
                    "Reason": f"IQR {multiplier:.1f}× 기준 밖",
                })

    return flagged, pd.DataFrame(details)


def build_batch_excel(
    result_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    *,
    all_result_df: pd.DataFrame | None = None,
    outlier_df: pd.DataFrame | None = None,
    W_um: float,
    L_um: float,
    eps_r: float,
    tox_nm: float,
    eps0: float,
    smoothing: int,
    ss_window: int,
    constant_current_A: float,
    col_vg: str,
    col_id: str,
    col_ig: str,
    col_vd: str,
) -> bytes:
    output = io.BytesIO()

    if all_result_df is None:
        all_result_df = result_df.copy()
    if outlier_df is None:
        outlier_df = pd.DataFrame()

    average_row = {"Device": "AVERAGE", "File": ""}
    for col in result_df.columns:
        if col in ["Device", "File", "Sheet"]:
            continue
        numeric = pd.to_numeric(result_df[col], errors="coerce")
        numeric = numeric[np.isfinite(numeric)]
        if len(numeric):
            average_row[col] = float(numeric.mean())

    summary_df = pd.concat([result_df, pd.DataFrame([average_row])], ignore_index=True)
    # Sheet/매칭 전류는 개별 결과표에서 제거하고, 평균은 별도 Averages 시트에 표시합니다.
    visible_excel_cols = [
        "Device", "File", "Vth F (V)", "Vth R (V)", "Vth Hys (V)",
        "gm_max (S)", "SS (mV/dec)", "muFE (cm2/Vs)", "ON/OFF",
    ]
    summary_df = summary_df[visible_excel_cols]
    averages_df = make_batch_style_summary(result_df)
    stats_df = statistics_dataframe(result_df)
    Cox = eps0 * eps_r / (tox_nm * 1e-7)

    params_df = pd.DataFrame([
        {"Parameter": "W", "Value": W_um, "Unit": "um"},
        {"Parameter": "L", "Value": L_um, "Unit": "um"},
        {"Parameter": "eps_r", "Value": eps_r, "Unit": "-"},
        {"Parameter": "tox", "Value": tox_nm, "Unit": "nm"},
        {"Parameter": "eps0", "Value": eps0, "Unit": "F/cm"},
        {"Parameter": "Cox", "Value": Cox, "Unit": "F/cm2"},
        {"Parameter": "Smoothing window", "Value": smoothing, "Unit": "points"},
        {"Parameter": "SS window", "Value": ss_window, "Unit": "points"},
        {"Parameter": "Constant current Iref", "Value": constant_current_A, "Unit": "A"},
        {"Parameter": "Vg column", "Value": col_vg, "Unit": ""},
        {"Parameter": "Id column", "Value": col_id, "Unit": ""},
        {"Parameter": "Ig column", "Value": col_ig, "Unit": ""},
        {"Parameter": "Vd column", "Value": col_vd, "Unit": ""},
    ])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary/Statistics는 이상치 제외가 적용된 데이터 기준
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        averages_df.to_excel(writer, sheet_name="Averages", index=False)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        # 원본 Batch 계산 결과는 항상 별도 보존
        all_result_df.drop(columns=["Sheet", "Vth match current (A)"], errors="ignore").to_excel(
            writer, sheet_name="All_Results", index=False
        )

        if not outlier_df.empty:
            outlier_df.to_excel(writer, sheet_name="Outliers", index=False)

        params_df.to_excel(writer, sheet_name="Parameters", index=False)
        if not errors_df.empty:
            errors_df.to_excel(writer, sheet_name="Errors", index=False)

        for ws in writer.book.worksheets:
            for column_cells in ws.columns:
                max_length = 0
                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))
                width = min(max(max_length + 2, 10), 35)
                ws.column_dimensions[column_cells[0].column_letter].width = width
            ws.freeze_panes = "A2"

    return output.getvalue()


def make_batch_style_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """Single/Batch 공통 평균표: Match current와 VDS는 평균에서만 표시합니다."""
    numeric_cols = [
        "Vth F (V)", "Vth R (V)", "Vth Hys (V)",
        "Vth match current (A)", "VDS (V)",
        "SS (mV/dec)", "muFE (cm2/Vs)", "ON/OFF",
    ]
    avg = result_df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()

    def mean_row(name, value, unit, decimals=3, scientific=False):
        rendered = fmt_sci(value) if scientific else fmt_num(value, decimals)
        return {
            "특성값": name,
            "Forward 평균": rendered,
            "Reverse 평균": "—",
            "Hysteresis |R-F|": "—",
            "단위": unit,
        }

    return pd.DataFrame([
        {
            "특성값": "Vth (F gm-max / R same-current)",
            "Forward 평균": fmt_num(avg["Vth F (V)"], 4),
            "Reverse 평균": fmt_num(avg["Vth R (V)"], 4),
            "Hysteresis |R-F|": fmt_num(avg["Vth Hys (V)"], 4),
            "단위": "V",
        },
        mean_row("Vth match current", avg["Vth match current (A)"], "A", scientific=True),
        mean_row("VDS", avg["VDS (V)"], "V", decimals=3),
        mean_row("SS", avg["SS (mV/dec)"], "mV/dec", decimals=2),
        mean_row("μFE", avg["muFE (cm2/Vs)"], "cm²/V·s", decimals=3),
        mean_row("ON/OFF", avg["ON/OFF"], "-", scientific=True),
    ])


def format_batch_style_results(result_df_with_status: pd.DataFrame) -> pd.DataFrame:
    """Single/Batch 공통 결과표: 요청한 9개 열만 표시합니다."""
    display_cols = [
        "Device", "사용 여부", "Vth F (V)", "Vth R (V)",
        "Vth Hys (V)", "gm_max (S)", "SS (mV/dec)",
        "muFE (cm2/Vs)", "ON/OFF",
    ]
    display_df = result_df_with_status[display_cols].copy()
    for col in ("gm_max (S)", "ON/OFF"):
        display_df[col] = display_df[col].apply(fmt_sci)

    fixed_formats = {
        "Vth F (V)": 4,
        "Vth R (V)": 4,
        "Vth Hys (V)": 4,
        "SS (mV/dec)": 2,
        "muFE (cm2/Vs)": 3,
    }
    for col, decimals in fixed_formats.items():
        display_df[col] = display_df[col].apply(lambda x, d=decimals: fmt_num(x, d))
    return display_df

def overlay_curve_record(row: dict, frames: dict) -> dict:
    """Batch/Single 공통 플롯 자료: Vth 매칭 점도 동일 방식으로 전달."""
    return {
        "Device": row["Device"],
        "Forward": frames["Forward"],
        "Reverse": frames["Reverse"],
        "Vg_gmmax_F": row["Vg@gmmax F (V)"],
        "gmmax_F": row["gm_max (S)"],
        "Vth_F": row["Vth F (V)"],
        "Vth_R": row["Vth R (V)"],
        "Vth_match_current": row["Vth match current (A)"],
    }


# ============================================================
# Session state
# ============================================================
for key, default in {
    "single_payload": None,
    "single_uploader_version": 0,
    "batch_payload": None,
    "batch_uploader_version": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# 버전 변경 뒤 기존 Streamlit 세션의 구 결과표/그림 캐시 사용 방지
ANALYSIS_SCHEMA_VERSION = 3
for payload_key in ("single_payload", "batch_payload"):
    payload = st.session_state[payload_key]
    if payload is not None and payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        st.session_state[payload_key] = None


# ============================================================
# UI
# ============================================================
# 화면 표시 및 다운로드 PNG에 동일하게 적용되는 그래프 가로:세로 비율
# 입력 비율을 유지한 채 종이 면적을 48 in² 부근으로 맞춥니다.
# 기본 4:3에서는 기존과 같은 8 × 6 inch 그래프가 생성됩니다.
def graph_figsize_from_ratio(width_ratio: float, height_ratio: float) -> tuple[float, float]:
    if not (np.isfinite(width_ratio) and np.isfinite(height_ratio)):
        raise ValueError("그래프 비율은 유효한 숫자여야 합니다.")
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError("그래프 가로와 세로 비율은 모두 0보다 커야 합니다.")

    ratio = float(width_ratio) / float(height_ratio)
    fig_width = float(np.sqrt(48.0 * ratio))
    fig_height = float(np.sqrt(48.0 / ratio))
    # 아주 극단적인 비율을 입력해도 PNG 크기가 지나치게 커지지 않게 제한합니다.
    scale = min(1.0, 12.0 / max(fig_width, fig_height))
    return fig_width * scale, fig_height * scale


st.markdown("**그래프 가로 : 세로 비율 (Single / Batch 공통)**")
ratio_width_col, ratio_height_col = st.columns(2)
with ratio_width_col:
    graph_ratio_width = st.number_input(
        "가로 비율",
        min_value=1,
        max_value=100,
        value=4,
        step=1,
        key="graph_ratio_width",
        help="예: 16:9 비율을 원하면 가로에 16을 입력하세요.",
    )
with ratio_height_col:
    graph_ratio_height = st.number_input(
        "세로 비율",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
        key="graph_ratio_height",
        help="예: 16:9 비율을 원하면 세로에 9를 입력하세요.",
    )

graph_figsize = graph_figsize_from_ratio(graph_ratio_width, graph_ratio_height)
st.caption(
    f"현재 그래프 비율: {graph_ratio_width}:{graph_ratio_height} "
    "| 기본값 4:3 | 숫자를 바꾸면 화면 그래프와 다운로드 PNG에 함께 반영됩니다."
)

tab_single, tab_batch = st.tabs(["Single Analyzer", "Batch Analyzer"])


# ------------------------- Single ----------------------------
with tab_single:
    st.subheader("Single Transistor Analyzer")

    upload_col, clear_col = st.columns([5, 1])
    with upload_col:
        single_file = st.file_uploader(
            "TFT 측정 Excel 파일 1개 업로드",
            type=["xls", "xlsx"],
            accept_multiple_files=False,
            key=f"single_file_{st.session_state.single_uploader_version}",
        )
    with clear_col:
        st.write("")
        st.write("")
        if st.button("파일 삭제", key="single_clear_all_files", use_container_width=True):
            st.session_state.single_uploader_version += 1
            st.session_state.single_payload = None
            st.session_state.pop("single_variation_metric", None)
            st.rerun()

    # 배치와 동일한 설정값 배치/순서 및 기본값.
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        single_smooth = st.number_input("Smoothing", min_value=1, max_value=31, value=5, step=2, key="single_smooth")
        single_ss = st.number_input("SS window", min_value=3, max_value=31, value=7, step=2, key="single_ss")
        single_iref = st.number_input(
            "Constant current Iref (A)", min_value=1e-15, value=1e-6,
            format="%.1e", key="single_iref",
        )
    with s2:
        single_w = st.number_input("W (µm)", value=100.0, key="single_w")
        single_l = st.number_input("L (µm)", value=10.0, key="single_l")
    with s3:
        single_epsr = st.number_input("εr", value=7.5, key="single_epsr")
        single_tox = st.number_input("tox (nm)", value=35.0, key="single_tox")
        single_eps0 = st.number_input("ε0 (F/cm)", value=8.85e-14, format="%.3e", key="single_eps0")
        single_abs = st.checkbox("분석에는 |Id| 사용", value=True, key="single_abs")
    with s4:
        single_vg = st.text_input("Vg 열", "BV", key="single_vg")
        single_id = st.text_input("Id 열", "AI", key="single_id")
        single_ig = st.text_input("Ig 열", "BI", key="single_ig")
        single_vd = st.text_input("Vd 열", "AV", key="single_vd")

    if single_file is not None:
        st.caption(
            "Batch와 동일하게 Vg/Id 열이 있는 가장 마지막 측정 Sheet를 자동 선택합니다. "
            "결과표/그래프/Excel 형식도 Batch와 동일합니다."
        )
        # 파일이 바뀌면 이전 데이터 결과를 잘못 표시하지 않도록 합니다.
        current_fingerprint = (
            single_file.name, single_file.size,
            hashlib.sha256(single_file.getvalue()).hexdigest(),
        )
        if st.session_state.single_payload is not None and st.session_state.single_payload.get("fingerprint") != current_fingerprint:
            st.session_state.single_payload = None
    else:
        st.session_state.single_payload = None
        current_fingerprint = None

    if st.button("Single 분석 실행", type="primary", key="single_run"):
        if single_file is None:
            st.warning("먼저 분석할 Excel 파일 1개를 업로드하세요.")
        else:
            try:
                with st.spinner("분석 중..."):
                    file_bytes = single_file.getvalue()
                    sheet = find_default_measurement_sheet(
                        file_bytes, single_vg.strip(), single_id.strip()
                    )
                    row, frames = analyze_batch_file(
                        file_bytes, single_file.name, sheet,
                        col_vg=single_vg.strip(), col_id=single_id.strip(),
                        col_ig=single_ig.strip(), col_vd=single_vd.strip(),
                        smoothing_window=int(single_smooth), ss_window=int(single_ss),
                        constant_current_A=float(single_iref),
                        use_abs_id=single_abs, W_um=float(single_w),
                        L_um=float(single_l), eps_r=float(single_epsr),
                        tox_nm=float(single_tox), eps0=float(single_eps0),
                    )
                    result_df = pd.DataFrame([row])
                    curves = {
                        f"001::{single_file.name}": overlay_curve_record(row, frames)
                    }
                    figs = make_batch_figures(curves, figsize=graph_figsize)
                    excel_bytes = build_batch_excel(
                        result_df, pd.DataFrame(),
                        W_um=float(single_w), L_um=float(single_l),
                        eps_r=float(single_epsr), tox_nm=float(single_tox),
                        eps0=float(single_eps0), smoothing=int(single_smooth),
                        ss_window=int(single_ss),
                        constant_current_A=float(single_iref),
                        col_vg=single_vg.strip(), col_id=single_id.strip(),
                        col_ig=single_ig.strip(), col_vd=single_vd.strip(),
                    )
                    stem = Path(single_file.name).stem
                    graph_zip = build_figures_zip(
                        figs,
                        [
                            f"{stem}_Linear_Id_Overlay.png",
                            f"{stem}_Log_Id_Ig_Overlay.png",
                            f"{stem}_gm_Forward.png",
                        ],
                        dpi=300,
                    )
                    st.session_state.single_payload = {
                        "schema_version": ANALYSIS_SCHEMA_VERSION,
                        "fingerprint": current_fingerprint,
                        "results": result_df,
                        "detected": pd.DataFrame([{
                            "Device": row["Device"], "File": single_file.name,
                            "Sheet": sheet, "Status": "Done",
                        }]),
                        "figs": figs,
                        "curves": curves,
                        "graph_figsize": graph_figsize,
                        "excel": excel_bytes,
                        "graph_zip": graph_zip,
                        "file_name": single_file.name,
                    }
            except Exception as e:
                st.session_state.single_payload = None
                st.exception(e)

    single_payload = st.session_state.single_payload
    if single_payload is not None:
        # 그래프 비율 변경 시 기존 분석값은 유지하고 플롯/PNG만 다시 생성합니다.
        if single_payload.get("graph_figsize") != graph_figsize:
            single_payload["figs"] = make_batch_figures(
                single_payload["curves"], figsize=graph_figsize
            )
            stem = Path(single_payload["file_name"]).stem
            single_payload["graph_zip"] = build_figures_zip(
                single_payload["figs"],
                [
                    f"{stem}_Linear_Id_Overlay.png",
                    f"{stem}_Log_Id_Ig_Overlay.png",
                    f"{stem}_gm_Forward.png",
                ],
                dpi=300,
            )
            single_payload["graph_figsize"] = graph_figsize

        result_df = single_payload["results"].copy()
        st.success("Single 완료 | 성공 1개 / 실패 0개")

        with st.expander("파일별 자동 선택 Sheet 확인"):
            st.dataframe(single_payload["detected"], use_container_width=True, hide_index=True)

        st.caption("통계 사용 소자: 전체 1개 / 사용 1개 / 제외 0개 (N=1)")
        # Batch와 완전히 같은 표 (단, 데이터 수 1개이므로 평균 = 그 소자 값).
        st.markdown("#### Single 특성값 (Batch 평균값과 동일한 표 형식, N=1)")
        st.dataframe(
            make_batch_style_summary(result_df),
            use_container_width=True, hide_index=True,
        )

        status_df = result_df.copy()
        status_df.insert(1, "사용 여부", "사용")
        st.dataframe(
            format_batch_style_results(status_df),
            use_container_width=True, hide_index=True,
        )

        with st.expander("특성값 분산 / 균일도 보기", expanded=False):
            st.info("소자가 1개이므로 소자 간 표준편차·분산·CV 및 이상치 판정은 불가능합니다. Batch에서 여러 소자를 분석해 주세요.")
            st.dataframe(
                format_statistics_for_display(statistics_dataframe(result_df)),
                use_container_width=True, hide_index=True,
            )
            variation_metric = st.selectbox(
                "분산 그래프로 볼 특성값",
                options=list(VARIABILITY_METRICS.keys()),
                key="single_variation_metric",
            )
            variation_fig = make_variability_figure(
                result_df, variation_metric, figsize=graph_figsize
            )
            display_centered_figure(variation_fig)
            variation_png = figure_to_png_bytes(variation_fig, dpi=300)
            safe_metric_name = variation_metric.replace("/", "_").replace(" ", "_").replace("μ", "mu")
            st.download_button(
                "분산 그래프 이미지 다운로드",
                data=variation_png,
                file_name=f"TFT_Single_Variation_{safe_metric_name}.png",
                mime="image/png", key="single_variation_download",
            )

        st.caption(
            "gm, SS, Mobility, ON/OFF는 Forward만 분석합니다. "
            "Reverse는 같은 전류의 Vth와 Hysteresis 계산에만 사용합니다. "
            "Log 그래프의 F gm-max Vth(○)와 R same-current Vth(◆)는 같은 |Id|에서 점선으로 연결됩니다."
        )
        sp1, sp2, sp3 = st.tabs([
            "Linear Id-Vg", "Log Id+Ig-Vg", "gm-Vg Forward",
        ])
        with sp1:
            display_centered_figure(single_payload["figs"][0])
        with sp2:
            display_centered_figure(single_payload["figs"][1])
        with sp3:
            display_centered_figure(single_payload["figs"][2])

        stem = Path(single_payload["file_name"]).stem
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Single 결과 Excel 다운로드",
                data=single_payload["excel"],
                file_name=f"{stem}_TFT_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="single_download", use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Single 그래프 이미지 다운로드",
                data=single_payload["graph_zip"],
                file_name=f"{stem}_TFT_Graphs.zip",
                mime="application/zip", key="single_graph_download",
                use_container_width=True,
            )


# ------------------------- Batch -----------------------------
with tab_batch:
    st.subheader("Batch Transistor Analyzer")

    upload_col, clear_col = st.columns([5, 1])

    with upload_col:
        batch_files = st.file_uploader(
            "여러 TFT 측정 Excel 파일 업로드",
            type=["xls", "xlsx"],
            accept_multiple_files=True,
            key=f"batch_files_{st.session_state.batch_uploader_version}",
        )

    with clear_col:
        st.write("")
        st.write("")
        if st.button(
            "파일 모두 삭제",
            key="batch_clear_all_files",
            use_container_width=True,
        ):
            # file_uploader는 값을 직접 비울 수 없으므로 key를 새로 만들어 초기화합니다.
            st.session_state.batch_uploader_version += 1
            st.session_state.batch_payload = None

            # 이전 Batch 분석/이상치 선택 상태도 함께 초기화합니다.
            for state_key in [
                "batch_outlier_mode",
                "batch_outlier_metrics",
                "batch_iqr_multiplier",
                "batch_manual_outliers",
                "batch_variation_metric",
            ]:
                st.session_state.pop(state_key, None)

            st.rerun()

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        batch_smooth = st.number_input("Smoothing", min_value=1, max_value=31, value=5, step=2, key="batch_smooth")
        batch_ss = st.number_input("SS window", min_value=3, max_value=31, value=7, step=2, key="batch_ss")
        batch_iref = st.number_input(
            "Constant current Iref (A)",
            min_value=1e-15,
            value=1e-6,
            format="%.1e",
            key="batch_iref",
        )
    with b2:
        batch_w = st.number_input("W (µm)", value=100.0, key="batch_w")
        batch_l = st.number_input("L (µm)", value=10.0, key="batch_l")
    with b3:
        batch_epsr = st.number_input("εr", value=7.5, key="batch_epsr")
        batch_tox = st.number_input("tox (nm)", value=35.0, key="batch_tox")
        batch_eps0 = st.number_input("ε0 (F/cm)", value=8.85e-14, format="%.3e", key="batch_eps0")
        batch_abs = st.checkbox("분석에는 |Id| 사용", value=True, key="batch_abs")
    with b4:
        batch_vg = st.text_input("Vg 열", "BV", key="batch_vg")
        batch_id = st.text_input("Id 열", "AI", key="batch_id")
        batch_ig = st.text_input("Ig 열", "BI", key="batch_ig")
        batch_vd = st.text_input("Vd 열", "AV", key="batch_vd")

    if batch_files:
        st.caption(
            "각 파일에서 BV/AI가 있는 가장 마지막 측정 Sheet를 자동 선택합니다. "
            "파일명이 같아도 서로 다른 업로드는 별도 소자로 분석하며, 예: d1 / d1_2 / d1_3으로 구분합니다."
        )

    if st.button("Batch 분석 실행", type="primary", key="batch_run"):
        if not batch_files:
            st.warning("먼저 분석할 Excel 파일을 여러 개 업로드하세요.")
        else:
            rows = []
            errors = []
            curves = {}
            detected_sheets = []

            progress = st.progress(0, text="Batch 분석 준비 중...")
            total = len(batch_files)

            # 브라우저 업로드 객체는 서로 다른 폴더의 파일이라도 파일명이 같을 수 있습니다.
            # 파일명을 내부 ID로 사용하면 d1.xls + d1.xls가 같은 데이터로 덮어써질 수 있으므로
            # 업로드 순서에 따라 고유한 Device label을 만들어 사용합니다.
            device_name_counts = {}

            for i, uploaded in enumerate(batch_files, start=1):
                progress.progress((i - 1) / total, text=f"분석 중 {i}/{total}: {uploaded.name}")
                file_bytes = uploaded.getvalue()

                base_device = Path(uploaded.name).stem
                device_name_counts[base_device] = device_name_counts.get(base_device, 0) + 1
                occurrence = device_name_counts[base_device]
                device_label = base_device if occurrence == 1 else f"{base_device}_{occurrence}"
                unique_file_key = f"{i:03d}::{uploaded.name}"

                try:
                    sheet = find_default_measurement_sheet(
                        file_bytes,
                        batch_vg.strip(),
                        batch_id.strip(),
                    )
                    row, file_curves = analyze_batch_file(
                        file_bytes,
                        uploaded.name,
                        sheet,
                        device_label=device_label,
                        col_vg=batch_vg.strip(),
                        col_id=batch_id.strip(),
                        col_ig=batch_ig.strip(),
                        col_vd=batch_vd.strip(),
                        smoothing_window=int(batch_smooth),
                        ss_window=int(batch_ss),
                        constant_current_A=float(batch_iref),
                        use_abs_id=batch_abs,
                        W_um=float(batch_w),
                        L_um=float(batch_l),
                        eps_r=float(batch_epsr),
                        tox_nm=float(batch_tox),
                        eps0=float(batch_eps0),
                    )
                    rows.append(row)
                    detected_sheets.append({"Device": device_label, "File": uploaded.name, "Sheet": sheet, "Status": "Done"})
                    curves[unique_file_key] = overlay_curve_record(row, file_curves)
                except Exception as e:
                    errors.append({
                        "Device": device_label,
                        "File": uploaded.name,
                        "Sheet": "",
                        "Error": str(e),
                    })
                    detected_sheets.append({"Device": device_label, "File": uploaded.name, "Sheet": "", "Status": f"ERROR: {e}"})

                progress.progress(i / total, text=f"분석 중 {i}/{total}: {uploaded.name}")

            progress.empty()

            if rows:
                result_df = pd.DataFrame(rows)
                errors_df = pd.DataFrame(errors)
                figs = make_batch_figures(curves, figsize=graph_figsize)
                excel_bytes = build_batch_excel(
                    result_df,
                    errors_df,
                    W_um=float(batch_w),
                    L_um=float(batch_l),
                    eps_r=float(batch_epsr),
                    tox_nm=float(batch_tox),
                    eps0=float(batch_eps0),
                    smoothing=int(batch_smooth),
                    ss_window=int(batch_ss),
                    constant_current_A=float(batch_iref),
                    col_vg=batch_vg.strip(),
                    col_id=batch_id.strip(),
                    col_ig=batch_ig.strip(),
                    col_vd=batch_vd.strip(),
                )

                graph_zip_bytes = build_figures_zip(
                    figs,
                    [
                        "TFT_Batch_Linear_Id_Overlay.png",
                        "TFT_Batch_Log_Id_Ig_Overlay.png",
                        "TFT_Batch_gm_Forward.png",
                    ],
                    dpi=300,
                )

                st.session_state.batch_payload = {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "results": result_df,
                    "errors": errors_df,
                    "curves": curves,
                    "figs": figs,
                    "excel": excel_bytes,
                    "graph_zip": graph_zip_bytes,
                    "detected": pd.DataFrame(detected_sheets),
                }
            else:
                st.session_state.batch_payload = {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "results": pd.DataFrame(),
                    "errors": pd.DataFrame(errors),
                    "curves": {},
                    "figs": None,
                    "excel": None,
                    "graph_zip": None,
                    "detected": pd.DataFrame(detected_sheets),
                }

    batch_payload = st.session_state.batch_payload
    if batch_payload is not None:
        result_df = batch_payload["results"]
        errors_df = batch_payload["errors"]

        success = len(result_df)
        failed = len(errors_df)
        if success:
            st.success(f"Batch 완료 | 성공 {success}개 / 실패 {failed}개")
        else:
            st.error(f"Batch 분석 실패 | 성공 0개 / 실패 {failed}개")

        if not batch_payload["detected"].empty:
            with st.expander("파일별 자동 선택 Sheet 확인"):
                st.dataframe(batch_payload["detected"], use_container_width=True, hide_index=True)

        if success:
            # Schema v3: 이전 코드의 결과 캐시는 자동 무효화되므로 별도 열 이름 변환이 불필요합니다.
            result_df = result_df.copy()

            # ----------------------------------------------------
            # Outlier filtering (원본 결과는 삭제하지 않음)
            # ----------------------------------------------------
            with st.expander("이상치 제외 설정", expanded=False):
                st.caption(
                    "이 기능은 원본 측정값을 삭제하지 않고, 평균/통계/분산 그래프/Batch overlay에서만 "
                    "선택한 소자를 제외합니다. Excel에는 전체 결과(All_Results)와 제외 내역(Outliers)을 함께 저장합니다."
                )

                outlier_mode = st.radio(
                    "이상치 처리 방법",
                    ["사용 안 함", "IQR 자동", "직접 선택"],
                    horizontal=True,
                    key="batch_outlier_mode",
                )

                excluded_devices: set[str] = set()
                outlier_details = pd.DataFrame()

                if outlier_mode == "IQR 자동":
                    selected_outlier_labels = st.multiselect(
                        "이상치 판정에 사용할 특성값",
                        options=list(OUTLIER_METRICS.keys()),
                        default=["Vth F (gm-max)", "Vth R (same-current)"],
                        key="batch_outlier_metrics",
                    )
                    iqr_multiplier = st.number_input(
                        "IQR 배수",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.5,
                        step=0.1,
                        key="batch_iqr_multiplier",
                    )
                    selected_columns = [
                        OUTLIER_METRICS[label]
                        for label in selected_outlier_labels
                    ]
                    excluded_devices, outlier_details = detect_iqr_outliers(
                        result_df, selected_columns, float(iqr_multiplier)
                    )
                    st.caption(
                        "IQR 자동은 Q1−k×IQR ~ Q3+k×IQR 범위를 벗어난 소자를 표시합니다. "
                        "기본 k=1.5이며, 선택한 특성 중 하나라도 기준을 벗어나면 해당 소자를 제외합니다."
                    )

                elif outlier_mode == "직접 선택":
                    manual_devices = st.multiselect(
                        "평균/통계/그래프에서 제외할 소자",
                        options=result_df["Device"].astype(str).tolist(),
                        key="batch_manual_outliers",
                    )
                    excluded_devices = set(manual_devices)
                    if excluded_devices:
                        outlier_details = pd.DataFrame([
                            {
                                "Device": dev,
                                "Metric": "Manual",
                                "Value": np.nan,
                                "Lower bound": np.nan,
                                "Upper bound": np.nan,
                                "Reason": "사용자 직접 제외",
                            }
                            for dev in sorted(excluded_devices)
                        ])

                if excluded_devices:
                    st.warning(
                        f"제외 대상 {len(excluded_devices)}개: "
                        + ", ".join(sorted(excluded_devices))
                    )
                    if not outlier_details.empty:
                        st.dataframe(
                            outlier_details,
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.info("현재 제외되는 소자가 없습니다.")

            active_df = result_df[
                ~result_df["Device"].astype(str).isin(excluded_devices)
            ].copy()

            if active_df.empty:
                st.error("모든 소자가 제외되어 분석할 데이터가 없습니다. 이상치 설정을 변경하세요.")
                active_df = result_df.copy()
                excluded_devices = set()
                outlier_details = pd.DataFrame()

            st.caption(
                f"Batch 통계 사용 소자: 전체 {len(result_df)}개 / 사용 {len(active_df)}개 / 제외 {len(excluded_devices)}개"
            )

            # 이상치 제외 상태를 원본 결과표에서도 확인할 수 있게 표시
            result_df_with_status = result_df.copy()
            result_df_with_status.insert(
                1,
                "사용 여부",
                result_df_with_status["Device"].astype(str).apply(
                    lambda d: "제외" if d in excluded_devices else "사용"
                ),
            )

            if excluded_devices:
                st.markdown("#### Batch 평균값 (이상치 제외 후)")
            else:
                st.markdown("#### Batch 평균값")
            st.dataframe(
                make_batch_style_summary(active_df),
                use_container_width=True,
                hide_index=True,
            )

            batch_display_df = format_batch_style_results(result_df_with_status)

            # 이상치로 제외된 소자는 결과표 전체 행을 연한 빨간색으로 강조
            def highlight_excluded_row(row):
                if str(row.get("사용 여부", "")) == "제외":
                    return [
                        "background-color: #ffd9d9; color: #8b0000;"
                    ] * len(row)
                return [""] * len(row)

            batch_display_styled = batch_display_df.style.apply(
                highlight_excluded_row,
                axis=1,
            )

            st.dataframe(
                batch_display_styled,
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------
            # Device-to-device variation / uniformity
            # ----------------------------------------------------
            with st.expander("특성값 분산 / 균일도 보기", expanded=False):
                st.caption(
                    "Std는 표준편차, Variance는 분산, CV는 평균 대비 상대 편차입니다. "
                    "같은 조건의 소자 균일성을 비교할 때는 보통 Std와 CV가 작을수록 균일합니다."
                )

                stats_df = statistics_dataframe(active_df)
                stats_display_df = format_statistics_for_display(stats_df)
                st.dataframe(
                    stats_display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                selected_variation_metric = st.selectbox(
                    "분산 그래프로 볼 특성값",
                    options=list(VARIABILITY_METRICS.keys()),
                    key="batch_variation_metric",
                )

                variation_fig = make_variability_figure(
                    active_df, selected_variation_metric, figsize=graph_figsize
                )
                display_centered_figure(variation_fig)
                st.caption(
                    "실선+점 = 각 소자의 값, 점선 = 평균, 옅은 영역 = 평균 ± 1σ(표준편차)"
                )

                variation_png = figure_to_png_bytes(variation_fig, dpi=300)
                safe_metric_name = (
                    selected_variation_metric
                    .replace("/", "_")
                    .replace(" ", "_")
                    .replace("μ", "mu")
                )
                st.download_button(
                    "분산 그래프 이미지 다운로드",
                    data=variation_png,
                    file_name=f"TFT_Batch_Variation_{safe_metric_name}.png",
                    mime="image/png",
                    key="batch_variation_download",
                )

            st.caption(
                "gm, SS, Mobility, ON/OFF는 Forward만 계산합니다. "
                "Reverse는 같은 전류의 Vth와 Hysteresis 계산에만 사용합니다. "
                "Log 그래프에는 Forward gm-max Vth(○)와 동일 전류 Reverse Vth(◆)를 점선으로 연결했습니다."
            )

            active_device_names = set(active_df["Device"].astype(str))
            active_curves = {
                key: item
                for key, item in batch_payload["curves"].items()
                if str(item.get("Device", "")) in active_device_names
            }
            active_figs = make_batch_figures(active_curves, figsize=graph_figsize)

            bp1, bp2, bp3 = st.tabs([
                "Linear Id-Vg",
                "Log Id+Ig-Vg",
                "gm-Vg Forward",
            ])
            with bp1:
                display_centered_figure(active_figs[0])
            with bp2:
                display_centered_figure(active_figs[1])
            with bp3:
                display_centered_figure(active_figs[2])

            active_excel_bytes = build_batch_excel(
                active_df,
                errors_df,
                all_result_df=result_df,
                outlier_df=outlier_details,
                W_um=float(batch_w),
                L_um=float(batch_l),
                eps_r=float(batch_epsr),
                tox_nm=float(batch_tox),
                eps0=float(batch_eps0),
                smoothing=int(batch_smooth),
                ss_window=int(batch_ss),
                constant_current_A=float(batch_iref),
                col_vg=batch_vg.strip(),
                col_id=batch_id.strip(),
                col_ig=batch_ig.strip(),
                col_vd=batch_vd.strip(),
            )

            active_graph_zip = build_figures_zip(
                active_figs,
                [
                    "TFT_Batch_Linear_Id_Overlay.png",
                    "TFT_Batch_Log_Id_Ig_Overlay.png",
                    "TFT_Batch_gm_Forward.png",
                ],
                dpi=300,
            )

            bdl1, bdl2 = st.columns(2)

            with bdl1:
                st.download_button(
                    "Batch 결과 Excel 다운로드",
                    data=active_excel_bytes,
                    file_name="TFT_Batch_Analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="batch_download",
                    use_container_width=True,
                )

            with bdl2:
                st.download_button(
                    "Batch 그래프 이미지 다운로드",
                    data=active_graph_zip,
                    file_name="TFT_Batch_Graphs.zip",
                    mime="application/zip",
                    key="batch_graph_download",
                    use_container_width=True,
                )

        if failed:
            with st.expander(f"분석 실패 파일 {failed}개"):
                st.dataframe(errors_df, use_container_width=True, hide_index=True)


st.divider()
st.caption(
    "Calculation: gm = dId/dVg · Primary Vth: Forward=gm-max tangent, Reverse=same |Id| as Forward Vth · "
    "Vth(constant-current) = Vg at |Id| = Iref · "
    "SS = max positive slope of log10(|Id|)-Vg · "
    "μFE = (L/W)·gm_max/(Cox·Vd)"
)
