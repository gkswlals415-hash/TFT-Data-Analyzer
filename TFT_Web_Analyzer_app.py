# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


# ============================================================
# Page
# ============================================================
st.set_page_config(
    page_title="TFT Data Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("TFT Data Analyzer")
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
    gm_search_vmin: float | None = None,
    gm_search_vmax: float | None = None,
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

    # gm_max 탐색 Vg 범위를 선택적으로 제한합니다.
    # Reverse에서 예를 들어 -10 V ~ 8 V만 사용하면 10 V 부근 끝단 피크가 제외됩니다.
    if gm_search_vmin is not None:
        candidate = candidate[vg[candidate] >= float(gm_search_vmin)]
    if gm_search_vmax is not None:
        candidate = candidate[vg[candidate] <= float(gm_search_vmax)]

    if len(candidate) == 0:
        range_text = (
            f"{gm_search_vmin if gm_search_vmin is not None else '-∞'} ~ "
            f"{gm_search_vmax if gm_search_vmax is not None else '+∞'} V"
        )
        raise ValueError(f"gm_max 탐색 범위({range_text}) 안에 유효한 데이터가 없습니다.")

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
        "gm_Search_Vmin_V": gm_search_vmin if gm_search_vmin is not None else np.nan,
        "gm_Search_Vmax_V": gm_search_vmax if gm_search_vmax is not None else np.nan,
    }
    return result, data


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
    reverse_gm_search_start_v: float | None = None,
    reverse_gm_search_end_v: float | None = None,
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
        gm_vmin = reverse_gm_search_start_v if label == "Reverse" else None
        gm_vmax = reverse_gm_search_end_v if label == "Reverse" else None

        result, analyzed = analyze_sweep(
            seg,
            smoothing_window=smoothing_window,
            ss_window=ss_window,
            constant_current_A=constant_current_A,
            gm_search_vmin=gm_vmin,
            gm_search_vmax=gm_vmax,
        )
        result = add_mobility(
            result,
            analyzed,
            W_um=W_um,
            L_um=L_um,
            eps_r=eps_r,
            tox_nm=tox_nm,
            eps0=eps0,
        )
        results[label] = result
        frames[label] = analyzed

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
    reverse_gm_search_start_v: float = -10.0,
    reverse_gm_search_end_v: float | None = None,
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
        reverse_gm_search_start_v=reverse_gm_search_start_v,
        reverse_gm_search_end_v=reverse_gm_search_end_v,
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
    hysteresis_cc = (
        abs(r["Vth_CC_V"] - f["Vth_CC_V"])
        if np.isfinite(r["Vth_CC_V"]) and np.isfinite(f["Vth_CC_V"])
        else np.nan
    )

    path = Path(file_name)
    row = {
        # 같은 파일명(d1.xls 등)이 여러 번 업로드되어도 서로 다른 소자로 유지
        "Device": device_label if device_label else path.stem,
        "File": path.name,
        "Sheet": sheet_name,
        "Vth F (V)": f["Vth_V"],
        "Vth R (V)": r["Vth_V"],
        "Vth Hysteresis |R-F| (V)": hysteresis,
        "Vth CC F (V)": f["Vth_CC_V"],
        "Vth CC R (V)": r["Vth_CC_V"],
        "Vth CC Hysteresis |R-F| (V)": hysteresis_cc,
        "Constant Current Iref (A)": f["Constant_Current_A"],
        "Reverse gm search start (V)": reverse_gm_search_start_v,
        "Reverse gm search end (V)": reverse_gm_search_end_v if reverse_gm_search_end_v is not None else np.nan,
        "gm_max F (S)": f["gm_max_S"],
        "gm_max R (S)": r["gm_max_S"],
        "SS F (mV/dec)": f["SS_mV_dec"],
        "SS R (mV/dec)": r["SS_mV_dec"],
        "muFE F (cm2/Vs)": f["mu_FE_cm2_Vs"],
        "muFE R (cm2/Vs)": r["mu_FE_cm2_Vs"],
        "ON/OFF F": f["On_Off"],
        "ON/OFF R": r["On_Off"],
        "Vg step (V)": float(np.nanmedian([f["Vg_Step_V"], r["Vg_Step_V"]])),
        "Vg@gmmax F (V)": f["Vg_at_gmmax_V"],
        "Vg@gmmax R (V)": r["Vg_at_gmmax_V"],
        "Id@gmmax F (A)": f["Id_at_gmmax_A"],
        "Id@gmmax R (A)": r["Id_at_gmmax_A"],
        "Vd used F (V)": f["Vd_used_V"],
        "Vd used R (V)": r["Vd_used_V"],
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


def single_results_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for label, r in results.items():
        rows.append({
            "Sweep": label,
            "Vth gm-max (V)": r["Vth_V"],
            "Vth constant-current (V)": r["Vth_CC_V"],
            "Iref (A)": r["Constant_Current_A"],
            "gm_max (S)": r["gm_max_S"],
            "SS (mV/dec)": r["SS_mV_dec"],
            "μFE (cm²/V·s)": r["mu_FE_cm2_Vs"],
            "Vg @ gm_max (V)": r["Vg_at_gmmax_V"],
            "Id @ gm_max (A)": r["Id_at_gmmax_A"],
            "ON/OFF": r["On_Off"],
            "Vg step (V)": r["Vg_Step_V"],
            "Vd used (V)": r["Vd_used_V"],
            "Cox (F/cm²)": r["Cox_F_cm2"],
        })
    return pd.DataFrame(rows)


def make_single_figures(results: dict, frames: dict, use_abs_id: bool):
    # Linear
    fig1 = Figure(figsize=(9, 5.2), dpi=110)
    ax1 = fig1.add_subplot(111)
    for label, data in frames.items():
        r = results[label]
        ax1.plot(data["Vg"], data["Id_analysis"], label=f"{label} |Id|")
        x0 = r["Vg_at_gmmax_V"]
        y0 = r["Id_at_gmmax_A"]
        gm = r["gm_max_S"]
        xline = np.linspace(np.nanmin(data["Vg"]), np.nanmax(data["Vg"]), 200)
        yline = y0 + gm * (xline - x0)
        ax1.plot(xline, yline, linestyle="--", alpha=0.65, label=f"{label} gm-max tangent")
        ax1.scatter([x0], [y0], s=35)
        ax1.axvline(r["Vth_V"], linestyle=":", alpha=0.7)
    ax1.set_xlabel("Gate Voltage, Vg (V)")
    ax1.set_ylabel("|Drain Current|, |Id| (A)" if use_abs_id else "Drain Current, Id (A)")
    ax1.set_title("Transfer Curve - Linear Scale")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    fig1.tight_layout()

    # Log Id + Ig
    fig2 = Figure(figsize=(9, 5.2), dpi=110)
    ax2 = fig2.add_subplot(111)

    # Constant-current 기준선
    iref_plot = np.nan
    if results:
        first_result = next(iter(results.values()))
        iref_plot = first_result.get("Constant_Current_A", np.nan)
    if np.isfinite(iref_plot) and iref_plot > 0:
        ax2.axhline(
            iref_plot,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"Constant current Iref={iref_plot:.1e} A",
        )

    for label, data in frames.items():
        r = results[label]
        y = np.abs(data["Id_raw"].to_numpy(dtype=float))
        y[y <= 0] = np.nan

        # Id curve 색상을 SS max-slope marker에도 그대로 사용
        line_id, = ax2.semilogy(data["Vg"], y, label=f"{label} |Id|")

        # Log 그래프에서는 gm_max가 아니라
        # SS 계산에 사용된 가장 가파른 log10(Id)-Vg 구간의 중심점을 표시
        ss_x = r.get("SS_Vg_center_V", np.nan)
        if np.isfinite(ss_x):
            vg_arr = data["Vg"].to_numpy(dtype=float)
            finite_idx = np.where(
                np.isfinite(vg_arr) & np.isfinite(y) & (y > 0)
            )[0]

            if len(finite_idx):
                nearest_idx = finite_idx[
                    np.argmin(np.abs(vg_arr[finite_idx] - ss_x))
                ]
                ss_y = float(y[nearest_idx])
                ss_x_plot = float(vg_arr[nearest_idx])

                ax2.scatter(
                    [ss_x_plot], [ss_y],
                    s=55,
                    color=line_id.get_color(),
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=6,
                    label="_nolegend_",
                )

        # Constant-current Vth 교차점: X marker
        cc_x = r.get("Vth_CC_V", np.nan)
        cc_i = r.get("Constant_Current_A", np.nan)
        if np.isfinite(cc_x) and np.isfinite(cc_i) and cc_i > 0:
            ax2.scatter(
                [cc_x], [cc_i],
                s=65,
                marker="x",
                color=line_id.get_color(),
                linewidths=1.8,
                zorder=7,
                label="_nolegend_",
            )

        if "Ig_raw" in data.columns:
            yig = np.abs(data["Ig_raw"].to_numpy(dtype=float))
            yig[yig <= 0] = np.nan
            ax2.semilogy(data["Vg"], yig, linestyle="--", alpha=0.7, label=f"{label} |Ig|")
    ax2.set_xlabel("Gate Voltage, Vg (V)")
    ax2.set_ylabel("Current (A)")
    ax2.set_title("Transfer Curve - Log Scale")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    # gm
    fig3 = Figure(figsize=(9, 5.2), dpi=110)
    ax3 = fig3.add_subplot(111)
    for label, data in frames.items():
        r = results[label]
        ax3.plot(data["Vg"], data["gm"], label=label)
        ax3.scatter([r["Vg_at_gmmax_V"]], [r["gm_max_S"]], s=40, label=f"{label} gm_max")
    ax3.set_xlabel("Gate Voltage, Vg (V)")
    ax3.set_ylabel("Transconductance, gm (S)")
    ax3.set_title("gm - Vg")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8)
    fig3.tight_layout()

    return fig1, fig2, fig3


def make_batch_figures(curves: dict):
    # Linear
    fig1 = Figure(figsize=(12, 6), dpi=100)
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
    fig2 = Figure(figsize=(12, 6), dpi=100)
    ax2 = fig2.add_subplot(111)
    device_handles = []
    any_ig = False

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
            ax2.semilogy(f["Vg"], ig_f, linestyle=":", linewidth=1.35, color=color, alpha=0.9, label="_nolegend_")
            ax2.semilogy(r["Vg"], ig_r, linestyle="-.", linewidth=1.35, color=color, alpha=0.9, label="_nolegend_")
            any_ig = True

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
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.35, label="Ig Forward"),
            Line2D([0], [0], color="black", linestyle="-.", linewidth=1.35, label="Ig Reverse"),
        ])
    ax2.legend(handles=style_handles, title="Current / Sweep", fontsize=8, loc="lower right")
    fig2.tight_layout()

    # gm - Forward only
    fig3 = Figure(figsize=(12, 6), dpi=100)
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

    # gm - Reverse only
    fig4 = Figure(figsize=(12, 6), dpi=100)
    ax4 = fig4.add_subplot(111)
    gm_r_handles = []

    for item in curves.values():
        device = item["Device"]
        r = item["Reverse"]

        line_r, = ax4.plot(
            r["Vg"], r["gm"], linestyle="--", linewidth=1.5, label=device
        )
        device_color = line_r.get_color()

        x_r = item.get("Vg_gmmax_R", np.nan)
        y_r = item.get("gmmax_R", np.nan)
        if np.isfinite(x_r) and np.isfinite(y_r):
            ax4.scatter(
                [x_r], [y_r],
                s=56, marker="s",
                color=device_color,
                edgecolors="black", linewidths=0.6,
                zorder=6, label="_nolegend_",
            )

        gm_r_handles.append(
            Line2D([0], [0], color=device_color, linewidth=1.8, linestyle="--", label=device)
        )

    ax4.set_xlabel("Gate Voltage, Vg (V)")
    ax4.set_ylabel("Transconductance, gm (S)")
    ax4.set_title("All Devices - Reverse gm-Vg")
    ax4.grid(True, alpha=0.25)

    if gm_r_handles:
        dev_leg_r = ax4.legend(
            handles=gm_r_handles,
            title="Device",
            fontsize=8,
            ncol=max(1, min(5, len(gm_r_handles))),
            loc="upper left",
        )
        ax4.add_artist(dev_leg_r)

    ax4.legend(
        handles=[
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1.5,
                marker="s", markersize=6, markeredgecolor="black",
                label="gm_max",
            )
        ],
        fontsize=8,
        loc="lower right",
    )
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4



def figure_to_png_bytes(fig: Figure, dpi: int = 300) -> bytes:
    """Matplotlib Figure를 고해상도 PNG bytes로 변환합니다."""
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight")
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


def build_single_excel(
    file_name: str,
    sheet_name: str,
    results: dict,
    frames: dict,
    raw_df: pd.DataFrame,
    *,
    W_um: float,
    L_um: float,
    eps_r: float,
    tox_nm: float,
    use_abs_id: bool,
) -> bytes:
    output = io.BytesIO()

    if "Forward" in results and "Reverse" in results:
        hys = abs(results["Reverse"]["Vth_V"] - results["Forward"]["Vth_V"])
    else:
        hys = np.nan

    summary_rows = []
    for label, r in results.items():
        summary_rows.append({
            "File": file_name,
            "Sheet": sheet_name,
            "Sweep": label,
            "Vth gm-max (V)": r["Vth_V"],
            "Vth constant-current (V)": r["Vth_CC_V"],
            "Constant current Iref (A)": r["Constant_Current_A"],
            "Mobility μFE (cm2/Vs)": r["mu_FE_cm2_Vs"],
            "Cox (F/cm2)": r["Cox_F_cm2"],
            "Vg step (V)": r["Vg_Step_V"],
            "Vd used (V)": r["Vd_used_V"],
            "W (um)": W_um,
            "L (um)": L_um,
            "eps_r": eps_r,
            "tox (nm)": tox_nm,
            "Vth Hysteresis |R-F| (V)": hys,
            "gm_max (S)": r["gm_max_S"],
            "Vg @ gm_max (V)": r["Vg_at_gmmax_V"],
            "Id @ gm_max (A)": r["Id_at_gmmax_A"],
            "SS (V/dec)": r["SS_V_dec"],
            "SS (mV/dec)": r["SS_mV_dec"],
            "SS center Vg (V)": r["SS_Vg_center_V"],
            "Ion (A)": r["Ion_A"],
            "Ioff (A)": r["Ioff_A"],
            "ON/OFF": r["On_Off"],
            "Smoothing Window": r["Smoothing_Window"],
            "SS Window": r["SS_Window"],
            "Use |Id|": use_abs_id,
        })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        raw_df.to_excel(writer, sheet_name="Raw_Read", index=False)
        for label, data in frames.items():
            export_df = data.copy()
            r = results[label]
            export_df["Vth_gmmax_result_V"] = np.nan
            export_df["Vth_constant_current_result_V"] = np.nan
            export_df["Constant_current_Iref_A"] = np.nan
            export_df["gm_max_result_S"] = np.nan
            export_df.loc[0, "Vth_gmmax_result_V"] = r["Vth_V"]
            export_df.loc[0, "Vth_constant_current_result_V"] = r["Vth_CC_V"]
            export_df.loc[0, "Constant_current_Iref_A"] = r["Constant_Current_A"]
            export_df.loc[0, "gm_max_result_S"] = r["gm_max_S"]
            export_df.to_excel(writer, sheet_name=f"{label}_Calc"[:31], index=False)

    return output.getvalue()


def statistics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Batch 특성값의 평균/표준편차/분산/CV/범위를 계산합니다."""
    metrics = [
        "Vth F (V)",
        "Vth R (V)",
        "Vth Hysteresis |R-F| (V)",
        "Vth CC F (V)",
        "Vth CC R (V)",
        "Vth CC Hysteresis |R-F| (V)",
        "gm_max F (S)",
        "gm_max R (S)",
        "SS F (mV/dec)",
        "SS R (mV/dec)",
        "muFE F (cm2/Vs)",
        "muFE R (cm2/Vs)",
        "ON/OFF F",
        "ON/OFF R",
        "Vg step (V)",
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
    "Vth (gm-max)": {
        "columns": ["Vth F (V)", "Vth R (V)"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "Vth (V)",
        "log": False,
    },
    "Vth (constant-current)": {
        "columns": ["Vth CC F (V)", "Vth CC R (V)"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "Vth CC (V)",
        "log": False,
    },
    "SS": {
        "columns": ["SS F (mV/dec)", "SS R (mV/dec)"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "SS (mV/dec)",
        "log": False,
    },
    "Mobility μFE": {
        "columns": ["muFE F (cm2/Vs)", "muFE R (cm2/Vs)"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "μFE (cm²/V·s)",
        "log": False,
    },
    "gm_max": {
        "columns": ["gm_max F (S)", "gm_max R (S)"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "gm_max (S)",
        "log": False,
    },
    "ON/OFF ratio": {
        "columns": ["ON/OFF F", "ON/OFF R"],
        "labels": ["Forward", "Reverse"],
        "ylabel": "ON/OFF ratio",
        "log": True,
    },
    "Vth Hysteresis (gm-max)": {
        "columns": ["Vth Hysteresis |R-F| (V)"],
        "labels": ["Hysteresis"],
        "ylabel": "Vth Hysteresis (V)",
        "log": False,
    },
    "Vth Hysteresis (constant-current)": {
        "columns": ["Vth CC Hysteresis |R-F| (V)"],
        "labels": ["Hysteresis"],
        "ylabel": "Vth CC Hysteresis (V)",
        "log": False,
    },
}


def make_variability_figure(df: pd.DataFrame, metric_name: str) -> Figure:
    """소자별 특성값과 평균 ± 1σ 범위를 함께 표시합니다."""
    info = VARIABILITY_METRICS[metric_name]
    columns = info["columns"]
    labels = info["labels"]

    fig = Figure(figsize=(10, 5.2), dpi=110)
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
    "Vth R (gm-max)": "Vth R (V)",
    "Vth Hysteresis (gm-max)": "Vth Hysteresis |R-F| (V)",
    "Vth CC F": "Vth CC F (V)",
    "Vth CC R": "Vth CC R (V)",
    "Vth CC Hysteresis": "Vth CC Hysteresis |R-F| (V)",
    "gm_max F": "gm_max F (S)",
    "gm_max R": "gm_max R (S)",
    "SS F": "SS F (mV/dec)",
    "SS R": "SS R (mV/dec)",
    "μFE F": "muFE F (cm2/Vs)",
    "μFE R": "muFE R (cm2/Vs)",
    "ON/OFF F": "ON/OFF F",
    "ON/OFF R": "ON/OFF R",
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
    reverse_gm_search_start_v: float,
    reverse_gm_search_end_v: float,
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

    average_row = {"Device": "AVERAGE", "File": "", "Sheet": f"N={len(result_df)}"}
    for col in result_df.columns:
        if col in ["Device", "File", "Sheet"]:
            continue
        numeric = pd.to_numeric(result_df[col], errors="coerce")
        numeric = numeric[np.isfinite(numeric)]
        if len(numeric):
            average_row[col] = float(numeric.mean())

    summary_df = pd.concat([result_df, pd.DataFrame([average_row])], ignore_index=True)
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
        {"Parameter": "Reverse gm search start", "Value": reverse_gm_search_start_v, "Unit": "V"},
        {"Parameter": "Reverse gm search end", "Value": reverse_gm_search_end_v, "Unit": "V"},
        {"Parameter": "Vg column", "Value": col_vg, "Unit": ""},
        {"Parameter": "Id column", "Value": col_id, "Unit": ""},
        {"Parameter": "Ig column", "Value": col_ig, "Unit": ""},
        {"Parameter": "Vd column", "Value": col_vd, "Unit": ""},
    ])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary/Statistics는 이상치 제외가 적용된 데이터 기준
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        # 원본 Batch 계산 결과는 항상 별도 보존
        all_result_df.to_excel(writer, sheet_name="All_Results", index=False)

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


# ============================================================
# Session state
# ============================================================
for key, default in {
    "single_payload": None,
    "batch_payload": None,
    "batch_uploader_version": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# UI
# ============================================================
tab_single, tab_batch = st.tabs(["Single Analyzer", "Batch Analyzer"])


# ------------------------- Single ----------------------------
with tab_single:
    st.subheader("Single TFT Analyzer")

    single_file = st.file_uploader(
        "TFT 측정 Excel 파일 업로드",
        type=["xls", "xlsx"],
        key="single_file",
    )

    if single_file is not None:
        file_bytes = single_file.getvalue()
        try:
            sheets = get_sheet_names(file_bytes)
        except Exception as e:
            st.error(f"Excel 파일을 읽을 수 없습니다: {e}")
            sheets = []
    else:
        sheets = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        single_sheet = st.selectbox("Sheet", sheets, index=0 if sheets else None, placeholder="파일을 먼저 업로드하세요")
        single_ss = st.number_input("SS window", min_value=3, max_value=31, value=7, step=2, key="single_ss")
        single_smooth = st.number_input("Smoothing window", min_value=1, max_value=31, value=5, step=2, key="single_smooth")
        single_iref = st.number_input(
            "Constant current Iref (A)",
            min_value=1e-15,
            value=1e-6,
            format="%.1e",
            key="single_iref",
        )
        single_reverse_gm_end = st.number_input(
            "Reverse gm_max 탐색 끝 Vg (V)",
            min_value=-10.0,
            max_value=10.0,
            value=8.0,
            step=0.1,
            format="%.1f",
            key="single_reverse_gm_end",
            help="Single 분석에서도 Reverse gm_max는 -10 V부터 이 값까지의 구간에서만 찾습니다. 예: 8.0 V이면 8~10 V 끝단 피크를 제외합니다.",
        )
    with c2:
        single_w = st.number_input("W (µm)", value=100.0, key="single_w")
        single_l = st.number_input("L (µm)", value=10.0, key="single_l")
        single_epsr = st.number_input("εr", value=7.5, key="single_epsr")
    with c3:
        single_tox = st.number_input("tox (nm)", value=35.0, key="single_tox")
        single_eps0 = st.number_input("ε0 (F/cm)", value=8.85e-14, format="%.3e", key="single_eps0")
        single_abs = st.checkbox("분석에는 |Id| 사용", value=True, key="single_abs")
    with c4:
        single_vg = st.text_input("Vg 열", "BV", key="single_vg")
        single_id = st.text_input("Id 열", "AI", key="single_id")
        single_ig = st.text_input("Ig 열", "BI", key="single_ig")
        single_vd = st.text_input("Vd 열", "AV", key="single_vd")

    st.caption(
        f"Single Reverse gm_max 탐색 범위: -10.0 V ~ {float(single_reverse_gm_end):.1f} V "
        "(Forward gm_max는 기존처럼 전체 sweep 범위에서 탐색)"
    )

    if st.button("분석 실행", type="primary", key="single_run"):
        if single_file is None:
            st.warning("먼저 Excel 파일을 업로드하세요.")
        elif not single_sheet:
            st.warning("분석할 Sheet를 선택하세요.")
        else:
            try:
                with st.spinner("분석 중..."):
                    results, frames, raw_df, header_row = analyze_single_file(
                        single_file.getvalue(),
                        single_sheet,
                        col_vg=single_vg.strip(),
                        col_id=single_id.strip(),
                        col_ig=single_ig.strip(),
                        col_vd=single_vd.strip(),
                        smoothing_window=int(single_smooth),
                        ss_window=int(single_ss),
                        constant_current_A=float(single_iref),
                        reverse_gm_search_start_v=-10.0,
                        reverse_gm_search_end_v=float(single_reverse_gm_end),
                        use_abs_id=single_abs,
                        W_um=float(single_w),
                        L_um=float(single_l),
                        eps_r=float(single_epsr),
                        tox_nm=float(single_tox),
                        eps0=float(single_eps0),
                    )
                    figs = make_single_figures(results, frames, single_abs)
                    excel_bytes = build_single_excel(
                        single_file.name,
                        single_sheet,
                        results,
                        frames,
                        raw_df,
                        W_um=float(single_w),
                        L_um=float(single_l),
                        eps_r=float(single_epsr),
                        tox_nm=float(single_tox),
                        use_abs_id=single_abs,
                    )

                    single_stem = Path(single_file.name).stem
                    graph_zip_bytes = build_figures_zip(
                        figs,
                        [
                            f"{single_stem}_Linear_Id_Vg.png",
                            f"{single_stem}_Log_Id_Ig_Vg.png",
                            f"{single_stem}_gm_Vg.png",
                        ],
                        dpi=300,
                    )

                    st.session_state.single_payload = {
                        "file_name": single_file.name,
                        "sheet": single_sheet,
                        "results": results,
                        "frames": frames,
                        "raw_df": raw_df,
                        "header_row": header_row,
                        "figs": figs,
                        "excel": excel_bytes,
                        "graph_zip": graph_zip_bytes,
                    }
            except Exception as e:
                st.session_state.single_payload = None
                st.exception(e)

    single_payload = st.session_state.single_payload
    if single_payload is not None:
        results = single_payload["results"]
        st.success(
            f"분석 완료 | Header: Excel {single_payload['header_row'] + 1}행 | "
            f"데이터 {len(single_payload['raw_df'])} points"
        )

        if "Forward" in results and "Reverse" in results:
            hys = abs(results["Reverse"]["Vth_V"] - results["Forward"]["Vth_V"])
            f = results["Forward"]
            r = results["Reverse"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Vth gm-max F", f"{f['Vth_V']:.4f} V")
            m2.metric("Vth gm-max R", f"{r['Vth_V']:.4f} V")
            m3.metric("gm-max Hysteresis |R-F|", f"{hys:.4f} V")
            m4.metric("SS F", f"{f['SS_mV_dec']:.2f} mV/dec" if np.isfinite(f['SS_mV_dec']) else "N/A")
            m5.metric("μFE F", f"{f['mu_FE_cm2_Vs']:.3f} cm²/V·s" if np.isfinite(f['mu_FE_cm2_Vs']) else "N/A")

            hys_cc = (
                abs(r["Vth_CC_V"] - f["Vth_CC_V"])
                if np.isfinite(r["Vth_CC_V"]) and np.isfinite(f["Vth_CC_V"])
                else np.nan
            )
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric(
                "Vth constant-current F",
                f"{f['Vth_CC_V']:.4f} V" if np.isfinite(f['Vth_CC_V']) else "N/A",
            )
            cc2.metric(
                "Vth constant-current R",
                f"{r['Vth_CC_V']:.4f} V" if np.isfinite(r['Vth_CC_V']) else "N/A",
            )
            cc3.metric(
                "CC Hysteresis |R-F|",
                f"{hys_cc:.4f} V" if np.isfinite(hys_cc) else "N/A",
            )

        # 화면 표시용: 작은/큰 값은 과학적 표기법으로 보기 쉽게 표시
        single_display_df = single_results_dataframe(results).copy()

        single_sci_cols = [
            "Iref (A)",
            "gm_max (S)",
            "Id @ gm_max (A)",
            "ON/OFF",
            "Cox (F/cm²)",
        ]
        for col in single_sci_cols:
            if col in single_display_df.columns:
                single_display_df[col] = single_display_df[col].apply(fmt_sci)

        single_fixed_formats = {
            "Vth gm-max (V)": 4,
            "Vth constant-current (V)": 4,
            "SS (mV/dec)": 2,
            "μFE (cm²/V·s)": 3,
            "Vg @ gm_max (V)": 3,
            "Vg step (V)": 3,
            "Vd used (V)": 3,
        }
        for col, decimals in single_fixed_formats.items():
            if col in single_display_df.columns:
                single_display_df[col] = single_display_df[col].apply(
                    lambda x, d=decimals: fmt_num(x, d)
                )

        st.dataframe(single_display_df, use_container_width=True, hide_index=True)

        p1, p2, p3 = st.tabs(["Linear Id-Vg", "Log Id+Ig-Vg", "gm-Vg"])
        with p1:
            display_centered_figure(single_payload["figs"][0])
        with p2:
            display_centered_figure(single_payload["figs"][1])
        with p3:
            display_centered_figure(single_payload["figs"][2])

        stem = Path(single_payload["file_name"]).stem
        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                "결과 Excel 다운로드",
                data=single_payload["excel"],
                file_name=f"{stem}_TFT_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="single_download",
                use_container_width=True,
            )

        with dl2:
            st.download_button(
                "그래프 이미지 다운로드",
                data=single_payload["graph_zip"],
                file_name=f"{stem}_TFT_graphs.zip",
                mime="application/zip",
                key="single_graph_download",
                use_container_width=True,
            )


# ------------------------- Batch -----------------------------
with tab_batch:
    st.subheader("Batch TFT Analyzer")

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
        batch_reverse_gm_end = st.number_input(
            "Reverse gm_max 탐색 끝 Vg (V)",
            min_value=-10.0,
            max_value=10.0,
            value=8.0,
            step=0.1,
            format="%.1f",
            key="batch_reverse_gm_end",
            help="Reverse gm_max는 -10 V부터 이 값까지의 구간에서만 찾습니다. 예: 8.0 V이면 8~10 V의 끝단 피크는 제외됩니다.",
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
                        reverse_gm_search_start_v=-10.0,
                        reverse_gm_search_end_v=float(batch_reverse_gm_end),
                        use_abs_id=batch_abs,
                        W_um=float(batch_w),
                        L_um=float(batch_l),
                        eps_r=float(batch_epsr),
                        tox_nm=float(batch_tox),
                        eps0=float(batch_eps0),
                    )
                    rows.append(row)
                    detected_sheets.append({"Device": device_label, "File": uploaded.name, "Sheet": sheet, "Status": "Done"})
                    curves[unique_file_key] = {
                        "Device": row["Device"],
                        "Forward": file_curves["Forward"],
                        "Reverse": file_curves["Reverse"],
                        "Vg_gmmax_F": row["Vg@gmmax F (V)"],
                        "gmmax_F": row["gm_max F (S)"],
                        "Vg_gmmax_R": row["Vg@gmmax R (V)"],
                        "gmmax_R": row["gm_max R (S)"],
                        "Reverse_gm_search_start": -10.0,
                        "Reverse_gm_search_end": float(batch_reverse_gm_end),
                    }
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
                figs = make_batch_figures(curves)
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
                    reverse_gm_search_start_v=-10.0,
                    reverse_gm_search_end_v=float(batch_reverse_gm_end),
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
                        "TFT_Batch_gm_Reverse.png",
                    ],
                    dpi=300,
                )

                st.session_state.batch_payload = {
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
            # ----------------------------------------------------
            # 이전 버전(v8.4 이하) session_state 결과와의 호환성 보정
            # v8.5에서 Hysteresis 열 이름을 |R-F| 표기로 변경했기 때문에
            # 기존 세션의 DataFrame에 새 열이 없으면 여기서 자동 생성합니다.
            # ----------------------------------------------------
            result_df = result_df.copy()

            hys_col = "Vth Hysteresis |R-F| (V)"
            old_hys_col = "Vth Hysteresis R-F (V)"
            if hys_col not in result_df.columns:
                if old_hys_col in result_df.columns:
                    result_df[hys_col] = pd.to_numeric(
                        result_df[old_hys_col], errors="coerce"
                    ).abs()
                elif "Vth F (V)" in result_df.columns and "Vth R (V)" in result_df.columns:
                    result_df[hys_col] = (
                        pd.to_numeric(result_df["Vth R (V)"], errors="coerce")
                        - pd.to_numeric(result_df["Vth F (V)"], errors="coerce")
                    ).abs()

            hys_cc_col = "Vth CC Hysteresis |R-F| (V)"
            old_hys_cc_col = "Vth CC Hysteresis R-F (V)"
            if hys_cc_col not in result_df.columns:
                if old_hys_cc_col in result_df.columns:
                    result_df[hys_cc_col] = pd.to_numeric(
                        result_df[old_hys_cc_col], errors="coerce"
                    ).abs()
                elif "Vth CC F (V)" in result_df.columns and "Vth CC R (V)" in result_df.columns:
                    result_df[hys_cc_col] = (
                        pd.to_numeric(result_df["Vth CC R (V)"], errors="coerce")
                        - pd.to_numeric(result_df["Vth CC F (V)"], errors="coerce")
                    ).abs()

            # 보정된 결과를 현재 세션에도 다시 저장
            batch_payload["results"] = result_df

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
                        default=["Vth F (gm-max)", "Vth R (gm-max)"],
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

            numeric_cols = [
                "Vth F (V)", "Vth R (V)", "Vth Hysteresis |R-F| (V)",
                "Vth CC F (V)", "Vth CC R (V)", "Vth CC Hysteresis |R-F| (V)",
                "gm_max F (S)", "gm_max R (S)",
                "SS F (mV/dec)", "SS R (mV/dec)",
                "muFE F (cm2/Vs)", "muFE R (cm2/Vs)",
                "ON/OFF F", "ON/OFF R",
                "Vg step (V)",
            ]
            avg = active_df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()

            # Batch 평균값을 한눈에 보기 쉬운 표로 표시
            if excluded_devices:
                st.markdown("#### Batch 평균값 (이상치 제외 후)")
            else:
                st.markdown("#### Batch 평균값")

            avg_table = pd.DataFrame([
                {
                    "특성값": "Vth (gm-max)",
                    "Forward 평균": f"{avg['Vth F (V)']:.4f}",
                    "Reverse 평균": f"{avg['Vth R (V)']:.4f}",
                    "Hysteresis |R-F|": f"{avg['Vth Hysteresis |R-F| (V)']:.4f}",
                    "단위": "V",
                },
                {
                    "특성값": "Vth (Constant Current)",
                    "Forward 평균": f"{avg['Vth CC F (V)']:.4f}",
                    "Reverse 평균": f"{avg['Vth CC R (V)']:.4f}",
                    "Hysteresis |R-F|": f"{avg['Vth CC Hysteresis |R-F| (V)']:.4f}",
                    "단위": "V",
                },
                {
                    "특성값": "SS",
                    "Forward 평균": f"{avg['SS F (mV/dec)']:.2f}",
                    "Reverse 평균": f"{avg['SS R (mV/dec)']:.2f}",
                    "Hysteresis |R-F|": "-",
                    "단위": "mV/dec",
                },
                {
                    "특성값": "μFE",
                    "Forward 평균": f"{avg['muFE F (cm2/Vs)']:.3f}",
                    "Reverse 평균": f"{avg['muFE R (cm2/Vs)']:.3f}",
                    "Hysteresis |R-F|": "-",
                    "단위": "cm²/V·s",
                },
                {
                    "특성값": "ON/OFF",
                    "Forward 평균": f"{avg['ON/OFF F']:.6e}",
                    "Reverse 평균": f"{avg['ON/OFF R']:.6e}",
                    "Hysteresis |R-F|": "-",
                    "단위": "-",
                },
            ])

            st.dataframe(
                avg_table,
                use_container_width=True,
                hide_index=True,
            )

            display_cols = [
                "Device", "사용 여부", "Sheet",
                "Vth F (V)", "Vth R (V)", "Vth Hysteresis |R-F| (V)",
                "Vth CC F (V)", "Vth CC R (V)", "Vth CC Hysteresis |R-F| (V)",
                "gm_max F (S)", "gm_max R (S)",
                "SS F (mV/dec)", "SS R (mV/dec)",
                "muFE F (cm2/Vs)", "muFE R (cm2/Vs)",
                "ON/OFF F", "ON/OFF R",
                "Vg step (V)",
            ]
            # 화면 표시용 표: gm / ON-OFF는 e 표기, 나머지는 필요한 소수점으로 정리
            batch_display_df = result_df_with_status[display_cols].copy()

            batch_sci_cols = [
                "gm_max F (S)",
                "gm_max R (S)",
                "ON/OFF F",
                "ON/OFF R",
            ]
            for col in batch_sci_cols:
                if col in batch_display_df.columns:
                    batch_display_df[col] = batch_display_df[col].apply(fmt_sci)

            batch_fixed_formats = {
                "Vth F (V)": 4,
                "Vth R (V)": 4,
                "Vth Hysteresis |R-F| (V)": 4,
                "Vth CC F (V)": 4,
                "Vth CC R (V)": 4,
                "Vth CC Hysteresis |R-F| (V)": 4,
                "SS F (mV/dec)": 2,
                "SS R (mV/dec)": 2,
                "muFE F (cm2/Vs)": 3,
                "muFE R (cm2/Vs)": 3,
                "Vg step (V)": 3,
            }
            for col, decimals in batch_fixed_formats.items():
                if col in batch_display_df.columns:
                    batch_display_df[col] = batch_display_df[col].apply(
                        lambda x, d=decimals: fmt_num(x, d)
                    )

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
                    active_df, selected_variation_metric
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
                f"Batch gm 그래프는 Forward와 Reverse를 분리해서 표시합니다. "
                f"Reverse gm_max는 -10.0 V ~ {float(batch_reverse_gm_end):.1f} V 범위에서만 탐색하며, "
                "각 그래프의 마커가 선택된 gm_max 위치입니다."
            )

            active_device_names = set(active_df["Device"].astype(str))
            active_curves = {
                key: item
                for key, item in batch_payload["curves"].items()
                if str(item.get("Device", "")) in active_device_names
            }
            active_figs = make_batch_figures(active_curves)

            bp1, bp2, bp3, bp4 = st.tabs([
                "Linear Id-Vg",
                "Log Id+Ig-Vg",
                "gm-Vg Forward",
                "gm-Vg Reverse",
            ])
            with bp1:
                display_centered_figure(active_figs[0])
            with bp2:
                display_centered_figure(active_figs[1])
            with bp3:
                display_centered_figure(active_figs[2])
            with bp4:
                display_centered_figure(active_figs[3])

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
                reverse_gm_search_start_v=-10.0,
                reverse_gm_search_end_v=float(batch_reverse_gm_end),
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
                    "TFT_Batch_gm_Reverse.png",
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
    "Calculation: gm = dId/dVg · Vth(gm-max) = gm-max tangent method · "
    "Vth(constant-current) = Vg at |Id| = Iref · "
    "SS = max positive slope of log10(|Id|)-Vg · "
    "μFE = (L/W)·gm_max/(Cox·Vd)"
)
