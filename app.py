import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="2群比較まとめ", layout="wide")
st.title("2群比較 統計検定アプリ")
st.caption("Shapiro-Wilk / Levene / Student t-test / Welch t-test / Mann-Whitney U を一括実行")

ALPHA_DEFAULT = 0.05


def load_csv_flex(uploaded_file):
    """UTF-8 / UTF-8-SIG / CP932 を順に試してCSVを読む"""
    raw = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "cp932", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


def to_numeric_series(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def summarize(x, name):
    x = pd.Series(x).dropna()
    q1 = x.quantile(0.25) if len(x) > 0 else np.nan
    q3 = x.quantile(0.75) if len(x) > 0 else np.nan
    return {
        "group": name,
        "n": int(x.shape[0]),
        "mean": float(x.mean()) if len(x) else np.nan,
        "sd": float(x.std(ddof=1)) if len(x) >= 2 else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "Q1": float(q1) if len(x) else np.nan,
        "Q3": float(q3) if len(x) else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }


def decision_text(p, alpha):
    if pd.isna(p):
        return "判定不可"
    return "有意差あり" if p < alpha else "有意差なし"


def add_result(results, test_name, statistic, pvalue, alpha, note=""):
    results.append({
        "test": test_name,
        "statistic": statistic,
        "p_value": pvalue,
        "alpha": alpha,
        "decision": decision_text(pvalue, alpha),
        "note": note,
    })


def safe_shapiro(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, np.nan, "n<3のため実行不可"
    res = stats.shapiro(x)
    return float(res.statistic), float(res.pvalue), ""


def run_tests(x, y, alpha=0.05):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    results = []

    sx_stat, sx_p, sx_note = safe_shapiro(x)
    sy_stat, sy_p, sy_note = safe_shapiro(y)
    add_result(results, "Shapiro-Wilk（群1）", sx_stat, sx_p, alpha, sx_note)
    add_result(results, "Shapiro-Wilk（群2）", sy_stat, sy_p, alpha, sy_note)

    if len(x) >= 2 and len(y) >= 2:
        lev = stats.levene(x, y, center="median")
        add_result(results, "Levene", float(lev.statistic), float(lev.pvalue), alpha, "")
    else:
        add_result(results, "Levene", np.nan, np.nan, alpha, "各群n>=2が必要")

    if len(x) >= 2 and len(y) >= 2:
        t_student = stats.ttest_ind(x, y, equal_var=True, alternative="two-sided", nan_policy="omit")
        add_result(results, "Student t-test", float(t_student.statistic), float(t_student.pvalue), alpha, "")
    else:
        add_result(results, "Student t-test", np.nan, np.nan, alpha, "各群n>=2が必要")

    if len(x) >= 2 and len(y) >= 2:
        t_welch = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided", nan_policy="omit")
        add_result(results, "Welch t-test", float(t_welch.statistic), float(t_welch.pvalue), alpha, "")
    else:
        add_result(results, "Welch t-test", np.nan, np.nan, alpha, "各群n>=2が必要")

    if len(x) >= 1 and len(y) >= 1:
        mw = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
        add_result(results, "Mann-Whitney U", float(mw.statistic), float(mw.pvalue), alpha, "")
    else:
        add_result(results, "Mann-Whitney U", np.nan, np.nan, alpha, "各群n>=1が必要")

    return pd.DataFrame(results)


st.sidebar.header("設定")
alpha = st.sidebar.number_input(
    "有意水準 α",
    min_value=0.001,
    max_value=0.20,
    value=ALPHA_DEFAULT,
    step=0.001,
    format="%.3f",
)

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv_flex(uploaded_file)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        st.stop()

    st.subheader("読み込んだデータ")
    st.dataframe(df.head(20), use_container_width=True)

    mode = st.radio(
        "データ形式を選択",
        ["long形式（群列1本＋値列1本）", "wide形式（各群が別列）"],
        horizontal=True,
    )

    if mode.startswith("long"):
        st.markdown("**想定例**: `group,value`")
        cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("群ラベル列", cols, index=0 if len(cols) > 0 else None)
        with col2:
            value_col = st.selectbox("数値列", cols, index=1 if len(cols) > 1 else 0)

        groups = df[group_col].dropna().astype(str).unique().tolist()

        if len(groups) < 2:
            st.error("群が2つ未満です。2群比較ができません。")
            st.stop()

        selected_groups = st.multiselect("比較する2群を選択", groups, default=groups[:2])

        if len(selected_groups) != 2:
            st.warning("比較する群をちょうど2つ選んでください。")
            st.stop()

        g1, g2 = selected_groups
        x = to_numeric_series(df.loc[df[group_col].astype(str) == g1, value_col])
        y = to_numeric_series(df.loc[df[group_col].astype(str) == g2, value_col])
        name1, name2 = g1, g2

    else:
        st.markdown("**想定例**: `group_A,group_B`")
        candidate_cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            col_x = st.selectbox("群1の列", candidate_cols, index=0 if len(candidate_cols) > 0 else None)
        with col2:
            default_index = 1 if len(candidate_cols) > 1 else 0
            col_y = st.selectbox("群2の列", candidate_cols, index=default_index)

        if col_x == col_y:
            st.warning("異なる2列を選んでください。")
            st.stop()

        x = to_numeric_series(df[col_x])
        y = to_numeric_series(df[col_y])
        name1, name2 = col_x, col_y

    st.subheader("記述統計")
    summary_df = pd.DataFrame([summarize(x, name1), summarize(y, name2)])
    st.dataframe(summary_df, use_container_width=True)

    if len(x) == 0 or len(y) == 0:
        st.error("どちらかの群で有効な数値データがありません。")
        st.stop()

    st.subheader("検定結果")
    results_df = run_tests(x, y, alpha=alpha)
    st.dataframe(results_df, use_container_width=True)

    st.subheader("簡易メモ")
    shapiro1 = results_df.loc[results_df["test"] == "Shapiro-Wilk（群1）", "p_value"].values[0]
    shapiro2 = results_df.loc[results_df["test"] == "Shapiro-Wilk（群2）", "p_value"].values[0]
    levene_p = results_df.loc[results_df["test"] == "Levene", "p_value"].values[0]

    shapiro_ok = pd.notna(shapiro1) and pd.notna(shapiro2) and shapiro1 >= alpha and shapiro2 >= alpha
    levene_ok = pd.notna(levene_p) and levene_p >= alpha

    if shapiro_ok and levene_ok:
        st.info("両群で正規性が大きく崩れておらず、等分散も大きく崩れていないので、Student t-test を確認しやすい状況です。")
    elif shapiro_ok and not levene_ok:
        st.info("正規性は大きく崩れていない一方、等分散性に注意が必要なので、Welch t-test を確認しやすい状況です。")
    else:
        st.info("正規性が十分でない可能性があるため、Mann-Whitney U もあわせて確認しやすい状況です。")

    csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="検定結果をCSVでダウンロード",
        data=csv_bytes,
        file_name="test_results.csv",
        mime="text/csv",
    )
else:
    st.info("CSVをアップロードしてください。")
