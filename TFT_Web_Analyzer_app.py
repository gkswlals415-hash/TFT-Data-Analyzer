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
        "Ion_A": ion,
        "Ioff_A": ioff,
        "On_Off": onoff,
        "Smoothing_Window": win,
        "SS_Window": ss_window,
        "gm_index": idx,
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
        result, analyzed = analyze_sweep(
            seg,
            smoothing_window=smoothing_window,
            ss_window=ss_window,
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
    col_vg: str,
    col_id: str,
    col_ig: str,
    col_vd: str,
    smoothing_window: int,
    ss_window: int,
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
    hysteresis = r["Vth_V"] - f["Vth_V"]

    path = Path(file_name)
    row = {
        "Device": path.stem,
        "File": path.name,
        "Sheet": sheet_name,
        "Vth F (V)": f["Vth_V"],
        "Vth R (V)": r["Vth_V"],
        "Vth Hysteresis R-F (V)": hysteresis,
        "gm_max F (S)": f["gm_max_S"],
        "gm_max R (S)": r["gm_max_S"],
        "SS F (mV/dec)": f["SS_mV_dec"],
        "SS R (mV/dec)": r["SS_mV_dec"],
        "muFE F (cm2/Vs)": f["mu_FE_cm2_Vs"],
        "muFE R (cm2/Vs)": r["mu_FE_cm2_Vs"],
        "ON/OFF F": f["On_Off"],
        "ON/OFF R": r["On_Off"],
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
            "Vth (V)": r["Vth_V"],
            "gm_max (S)": r["gm_max_S"],
            "SS (mV/dec)": r["SS_mV_dec"],
            "μFE (cm²/V·s)": r["mu_FE_cm2_Vs"],
            "Vg @ gm_max (V)": r["Vg_at_gmmax_V"],
            "Id @ gm_max (A)": r["Id_at_gmmax_A"],
            "ON/OFF": r["On_Off"],
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
    for label, data in frames.items():
        y = np.abs(data["Id_raw"].to_numpy(dtype=float))
        y[y <= 0] = np.nan
        ax2.semilogy(data["Vg"], y, label=f"{label} |Id|")
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

    # gm
    fig3 = Figure(figsize=(12, 6), dpi=100)
    ax3 = fig3.add_subplot(111)
    for item in curves.values():
        device = item["Device"]
        f = item["Forward"]
        r = item["Reverse"]
        line_f, = ax3.plot(f["Vg"], f["gm"], label=device, linewidth=1.5)
        ax3.plot(
            r["Vg"], r["gm"], linestyle="--", linewidth=1.2,
            color=line_f.get_color(), label="_nolegend_"
        )
    ax3.set_xlabel("Gate Voltage, Vg (V)")
    ax3.set_ylabel("Transconductance, gm (S)")
    ax3.set_title("All Devices - gm-Vg (solid F / dashed R)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(title="Device", fontsize=8, ncol=max(1, min(5, len(curves))))
    fig3.tight_layout()

    return fig1, fig2, fig3



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
        hys = results["Reverse"]["Vth_V"] - results["Forward"]["Vth_V"]
    else:
        hys = np.nan

    summary_rows = []
    for label, r in results.items():
        summary_rows.append({
            "File": file_name,
            "Sheet": sheet_name,
            "Sweep": label,
            "Vth (V)": r["Vth_V"],
            "Mobility μFE (cm2/Vs)": r["mu_FE_cm2_Vs"],
            "Cox (F/cm2)": r["Cox_F_cm2"],
            "Vd used (V)": r["Vd_used_V"],
            "W (um)": W_um,
            "L (um)": L_um,
            "eps_r": eps_r,
            "tox (nm)": tox_nm,
            "Vth Hysteresis R-F (V)": hys,
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
            export_df["Vth_result_V"] = np.nan
            export_df["gm_max_result_S"] = np.nan
            export_df.loc[0, "Vth_result_V"] = r["Vth_V"]
            export_df.loc[0, "gm_max_result_S"] = r["gm_max_S"]
            export_df.to_excel(writer, sheet_name=f"{label}_Calc"[:31], index=False)

    return output.getvalue()


def statistics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Vth F (V)",
        "Vth R (V)",
        "Vth Hysteresis R-F (V)",
        "gm_max F (S)",
        "gm_max R (S)",
        "SS F (mV/dec)",
        "SS R (mV/dec)",
        "muFE F (cm2/Vs)",
        "muFE R (cm2/Vs)",
        "ON/OFF F",
        "ON/OFF R",
    ]
    rows = []
    for metric in metrics:
        s = pd.to_numeric(df[metric], errors="coerce")
        s = s[np.isfinite(s)]
        rows.append({
            "Metric": metric,
            "N": int(s.count()),
            "Mean": float(s.mean()) if len(s) else np.nan,
            "Std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "Min": float(s.min()) if len(s) else np.nan,
            "Max": float(s.max()) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def build_batch_excel(
    result_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    *,
    W_um: float,
    L_um: float,
    eps_r: float,
    tox_nm: float,
    eps0: float,
    smoothing: int,
    ss_window: int,
    col_vg: str,
    col_id: str,
    col_ig: str,
    col_vd: str,
) -> bytes:
    output = io.BytesIO()

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
        {"Parameter": "Vg column", "Value": col_vg, "Unit": ""},
        {"Parameter": "Id column", "Value": col_id, "Unit": ""},
        {"Parameter": "Ig column", "Value": col_ig, "Unit": ""},
        {"Parameter": "Vd column", "Value": col_vd, "Unit": ""},
    ])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)
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
            hys = results["Reverse"]["Vth_V"] - results["Forward"]["Vth_V"]
            f = results["Forward"]
            r = results["Reverse"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Vth F", f"{f['Vth_V']:.4f} V")
            m2.metric("Vth R", f"{r['Vth_V']:.4f} V")
            m3.metric("Hysteresis R-F", f"{hys:.4f} V")
            m4.metric("SS F", f"{f['SS_mV_dec']:.2f} mV/dec" if np.isfinite(f['SS_mV_dec']) else "N/A")
            m5.metric("μFE F", f"{f['mu_FE_cm2_Vs']:.3f} cm²/V·s" if np.isfinite(f['mu_FE_cm2_Vs']) else "N/A")

        st.dataframe(single_results_dataframe(results), use_container_width=True, hide_index=True)

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

    batch_files = st.file_uploader(
        "여러 TFT 측정 Excel 파일 업로드",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        key="batch_files",
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        batch_smooth = st.number_input("Smoothing", min_value=1, max_value=31, value=5, step=2, key="batch_smooth")
        batch_ss = st.number_input("SS window", min_value=3, max_value=31, value=7, step=2, key="batch_ss")
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
        st.caption("각 파일에서 BV/AI가 있는 가장 마지막 측정 Sheet를 자동 선택합니다.")

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

            for i, uploaded in enumerate(batch_files, start=1):
                progress.progress((i - 1) / total, text=f"분석 중 {i}/{total}: {uploaded.name}")
                file_bytes = uploaded.getvalue()
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
                        col_vg=batch_vg.strip(),
                        col_id=batch_id.strip(),
                        col_ig=batch_ig.strip(),
                        col_vd=batch_vd.strip(),
                        smoothing_window=int(batch_smooth),
                        ss_window=int(batch_ss),
                        use_abs_id=batch_abs,
                        W_um=float(batch_w),
                        L_um=float(batch_l),
                        eps_r=float(batch_epsr),
                        tox_nm=float(batch_tox),
                        eps0=float(batch_eps0),
                    )
                    rows.append(row)
                    detected_sheets.append({"File": uploaded.name, "Sheet": sheet, "Status": "Done"})
                    curves[uploaded.name] = {
                        "Device": row["Device"],
                        "Forward": file_curves["Forward"],
                        "Reverse": file_curves["Reverse"],
                    }
                except Exception as e:
                    errors.append({
                        "Device": Path(uploaded.name).stem,
                        "File": uploaded.name,
                        "Sheet": "",
                        "Error": str(e),
                    })
                    detected_sheets.append({"File": uploaded.name, "Sheet": "", "Status": f"ERROR: {e}"})

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
                        "TFT_Batch_gm_Overlay.png",
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
            numeric_cols = [
                "Vth F (V)", "Vth R (V)", "Vth Hysteresis R-F (V)",
                "gm_max F (S)", "gm_max R (S)",
                "SS F (mV/dec)", "SS R (mV/dec)",
                "muFE F (cm2/Vs)", "muFE R (cm2/Vs)",
                "ON/OFF F", "ON/OFF R",
            ]
            avg = result_df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()

            st.markdown(
                "**평균 | "
                f"Vth F={avg['Vth F (V)']:.4f} V, "
                f"Vth R={avg['Vth R (V)']:.4f} V, "
                f"Hys={avg['Vth Hysteresis R-F (V)']:.4f} V, "
                f"SS F={avg['SS F (mV/dec)']:.2f} mV/dec, "
                f"SS R={avg['SS R (mV/dec)']:.2f} mV/dec, "
                f"μFE F={avg['muFE F (cm2/Vs)']:.3f}, "
                f"μFE R={avg['muFE R (cm2/Vs)']:.3f} cm²/V·s, "
                f"ON/OFF F={avg['ON/OFF F']:.6e}, "
                f"ON/OFF R={avg['ON/OFF R']:.6e}**"
            )

            display_cols = [
                "Device", "Sheet",
                "Vth F (V)", "Vth R (V)", "Vth Hysteresis R-F (V)",
                "gm_max F (S)", "gm_max R (S)",
                "SS F (mV/dec)", "SS R (mV/dec)",
                "muFE F (cm2/Vs)", "muFE R (cm2/Vs)",
                "ON/OFF F", "ON/OFF R",
            ]
            st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

            bp1, bp2, bp3 = st.tabs(["Linear Id-Vg", "Log Id+Ig-Vg", "gm-Vg"])
            with bp1:
                display_centered_figure(batch_payload["figs"][0])
            with bp2:
                display_centered_figure(batch_payload["figs"][1])
            with bp3:
                display_centered_figure(batch_payload["figs"][2])

            bdl1, bdl2 = st.columns(2)

            with bdl1:
                st.download_button(
                    "Batch 결과 Excel 다운로드",
                    data=batch_payload["excel"],
                    file_name="TFT_Batch_Analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="batch_download",
                    use_container_width=True,
                )

            with bdl2:
                st.download_button(
                    "Batch 그래프 이미지 다운로드",
                    data=batch_payload["graph_zip"],
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
    "Calculation: gm = dId/dVg · Vth = gm-max tangent method · "
    "SS = max positive slope of log10(|Id|)-Vg · "
    "μFE = (L/W)·gm_max/(Cox·Vd)"
)
