import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="独立2群比較 統計検定", layout="wide")
st.title("独立2群比較 統計検定")
st.caption("Shapiro-Wilk / Levene / Student t-test / Welch t-test / Mann-Whitney U を実行")
st.warning("このアプリケーションは独立2群用です。対応のあるデータ（前後比較・同一対象の2条件比較）には使用しないでください。")

ALPHA_DEFAULT = 0.05
SAMPLE_WIDE_CSV = """group_A,group_B,group_C
12.3,10.2,12.4
11.8,9.8,11.4
13.1,10.9,10.9
12.2,11.1,12.1
"""

def load_csv_flex(uploaded_file):
    """UTF-8 / UTF-8-SIG / CP932 を順に試してCSVを読む"""
    raw = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "cp932"]
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
        "グループ": name,
        "n数": int(x.shape[0]),
        "平均": float(x.mean()) if len(x) else np.nan,
        "標準偏差": float(x.std(ddof=1)) if len(x) >= 2 else np.nan,
        "最少値": float(x.min()) if len(x) else np.nan,
        "Q1": float(q1) if len(x) else np.nan,
        "中央値": float(x.median()) if len(x) else np.nan,
        "Q3": float(q3) if len(x) else np.nan,
        "最大値": float(x.max()) if len(x) else np.nan,
    }


def add_result(results, category, test_name, pvalue, alpha, interpretation, note="", primary=False):
    results.append({
        "区分": category,
        "検定": test_name,
        "p値": pvalue,
        "α": alpha,
        "推奨": "○" if primary else "",
        "解釈": interpretation,
        "備考": note if note else "特記事項なし",
    })


def safe_shapiro(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, "n<3のため実行不可"
    if len(x) > 5000:
        return np.nan, "n>5000のため Shapiro-Wilk は実行対象外（別法の検討推奨）"
    if x.nunique() < 2:
        return np.nan, "値のばらつきがほとんどないため実行不可"
    try:
        res = stats.shapiro(x)
        return float(res.pvalue), ""
    except Exception as e:
        return np.nan, f"実行不可: {e}"


def interpret_shapiro(p, alpha):
    if pd.isna(p):
        return "判定不可"
    if p < alpha:
        return "正規性を仮定しにくい"
    return "正規性を棄却する十分な根拠なし"


def interpret_levene(p, alpha):
    if pd.isna(p):
        return "判定不可"
    if p < alpha:
        return "等分散を仮定しにくい"
    return "等分散を棄却する十分な根拠なし"


def interpret_difference(p, alpha):
    if pd.isna(p):
        return "判定不可"
    if p < alpha:
        return "有意差あり"
    return "有意差を示す十分な根拠なし"


def choose_primary_test(x, y, shapiro1_p, shapiro2_p, levene_p, alpha):
    """
    推奨される検定を1つ返す。
    実務上のわかりやすさを優先した簡易ルール。
    """
    if len(x) < 2 or len(y) < 2:
        return "Mann-Whitney U"

    shapiro_ok = (
        pd.notna(shapiro1_p) and pd.notna(shapiro2_p)
        and shapiro1_p >= alpha and shapiro2_p >= alpha
    )
    levene_ok = pd.notna(levene_p) and levene_p >= alpha

    if shapiro_ok and levene_ok:
        return "Student t-test"
    if shapiro_ok and not levene_ok:
        return "Welch t-test"
    return "Mann-Whitney U"


def run_tests(x, y, alpha=0.05):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    results = []

    sx_p, sx_note = safe_shapiro(x)
    sy_p, sy_note = safe_shapiro(y)

    add_result(
        results, "前提確認", "Shapiro-Wilk（群1）",
        sx_p, alpha, interpret_shapiro(sx_p, alpha), sx_note
    )
    add_result(
        results, "前提確認", "Shapiro-Wilk（群2）",
        sy_p, alpha, interpret_shapiro(sy_p, alpha), sy_note
    )

    if len(x) >= 2 and len(y) >= 2 and x.nunique() >= 2 and y.nunique() >= 2:
        try:
            lev = stats.levene(x, y, center="median")
            lev_p = float(lev.pvalue)
            lev_note = ""
        except Exception as e:
            lev_p, lev_note = np.nan, f"実行不可: {e}"
    else:
        lev_p, lev_note = np.nan, "各群 n>=2 かつ一定値のみでないことが必要"

    add_result(
        results, "前提確認", "Levene",
        lev_p, alpha, interpret_levene(lev_p, alpha), lev_note
    )

    primary_test = choose_primary_test(x, y, sx_p, sy_p, lev_p, alpha)

    if len(x) >= 2 and len(y) >= 2:
        try:
            t_student = stats.ttest_ind(
                x, y, equal_var=True, alternative="two-sided", nan_policy="omit"
            )
            add_result(
                results, "群比較", "Student t-test",
                float(t_student.pvalue), alpha,
                interpret_difference(float(t_student.pvalue), alpha),
                primary=(primary_test == "Student t-test")
            )
        except Exception as e:
            add_result(
                results, "群比較", "Student t-test",
                np.nan, alpha, "判定不可", f"実行不可: {e}",
                primary=(primary_test == "Student t-test")
            )
    else:
        add_result(
            results, "群比較", "Student t-test",
            np.nan, alpha, "判定不可", "各群 n>=2 が必要",
            primary=(primary_test == "Student t-test")
        )

    if len(x) >= 2 and len(y) >= 2:
        try:
            t_welch = stats.ttest_ind(
                x, y, equal_var=False, alternative="two-sided", nan_policy="omit"
            )
            add_result(
                results, "群比較", "Welch t-test",
                float(t_welch.pvalue), alpha,
                interpret_difference(float(t_welch.pvalue), alpha),
                primary=(primary_test == "Welch t-test")
            )
        except Exception as e:
            add_result(
                results, "群比較", "Welch t-test",
                np.nan, alpha, "判定不可", f"実行不可: {e}",
                primary=(primary_test == "Welch t-test")
            )
    else:
        add_result(
            results, "群比較", "Welch t-test",
            np.nan, alpha, "判定不可", "各群 n>=2 が必要",
            primary=(primary_test == "Welch t-test")
        )

    if len(x) >= 1 and len(y) >= 1:
        try:
            mw = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
            add_result(
                results, "群比較", "Mann-Whitney U",
                float(mw.pvalue), alpha,
                interpret_difference(float(mw.pvalue), alpha),
                primary=(primary_test == "Mann-Whitney U")
            )
        except Exception as e:
            add_result(
                results, "群比較", "Mann-Whitney U",
                np.nan, alpha, "判定不可", f"実行不可: {e}",
                primary=(primary_test == "Mann-Whitney U")
            )
    else:
        add_result(
            results, "群比較", "Mann-Whitney U",
            np.nan, alpha, "判定不可", "各群 n>=1 が必要",
            primary=(primary_test == "Mann-Whitney U")
        )

    return pd.DataFrame(results), primary_test, sx_p, sy_p, lev_p


st.sidebar.header("設定")
alpha = st.sidebar.selectbox(
    "有意水準 α",
    options=[0.01, 0.05, 0.10],
    index=1,
    format_func=lambda x: f"{x:.2f}",
)

with st.expander("CSV形式の例", expanded=True):
    st.markdown("2列以上の数値列を持つCSVをアップロードしてください。比較は2列を選択して行います。")
    st.code(SAMPLE_WIDE_CSV, language="csv")
    st.download_button(
        label="サンプルCSVをダウンロード",
        data=SAMPLE_WIDE_CSV.encode("utf-8-sig"),
        file_name="sample_wide.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv_flex(uploaded_file)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        st.stop()

    if df.shape[1] < 2:
        st.error("列が2本以上必要です。")
        st.stop()

    st.subheader("読み込んだデータ")
    st.dataframe(df.head(20), use_container_width=True)

    candidate_cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        col_x = st.selectbox("群1の列", candidate_cols, index=0)
    with col2:
        col_y = st.selectbox("群2の列", candidate_cols, index=1)

    if col_x == col_y:
        st.warning("異なる2列を選んでください。")
        st.stop()

    x_raw = df[col_x]
    y_raw = df[col_y]
    x = to_numeric_series(x_raw)
    y = to_numeric_series(y_raw)

    dropped_x = int(x_raw.notna().sum() - len(x))
    dropped_y = int(y_raw.notna().sum() - len(y))
    if dropped_x > 0 or dropped_y > 0:
        st.warning(f"数値変換できない値を除外しました（群1: {dropped_x}件, 群2: {dropped_y}件）。")

    if len(x) == 0 or len(y) == 0:
        st.error("どちらかの群で有効な数値データがありません。")
        st.stop()

    if len(x) < 3 or len(y) < 3:
        st.info("Shapiro-Wilk は n<3 では実行できません。少数例では正規性判断に注意してください。")

    if len(x) > 5000 or len(y) > 5000:
        st.info("n>5000 の群では Shapiro-Wilk を省略します。大標本では図示や別法も検討してください。")

    if x.nunique() < 2 or y.nunique() < 2:
        st.warning("一方または両方の群で値のばらつきがほとんどありません。検定結果の解釈に注意してください。")

    st.subheader("記述統計")
    summary_df = pd.DataFrame([summarize(x, col_x), summarize(y, col_y)])
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("箱ひげ図")
    fig_box, ax_box = plt.subplots(figsize=(7, 5))
    ax_box.boxplot([x, y], labels=[col_x, col_y])
    ax_box.set_title("群ごとの箱ひげ図")
    ax_box.set_xlabel("群")
    ax_box.set_ylabel("値")
    ax_box.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig_box)

    st.subheader("QQプロット")
    fig_qq, axes = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(x, dist="norm", plot=axes[0])
    axes[0].set_title(f"{col_x} のQQプロット")
    axes[0].grid(True, alpha=0.3)

    stats.probplot(y, dist="norm", plot=axes[1])
    axes[1].set_title(f"{col_y} のQQプロット")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_qq)

    st.subheader("検定結果")
    results_df, primary_test, shapiro1, shapiro2, levene_p = run_tests(x, y, alpha=alpha)
    st.dataframe(results_df, use_container_width=True)

    st.subheader("解釈メモ")

    shapiro_ok = (
        pd.notna(shapiro1) and pd.notna(shapiro2)
        and shapiro1 >= alpha and shapiro2 >= alpha
    )
    levene_ok = pd.notna(levene_p) and levene_p >= alpha

    if shapiro_ok and levene_ok:
        st.info(
            "両群とも正規性を棄却する十分な根拠がなく、等分散性も棄却する十分な根拠がないため、"
            "この場合は Student t-test が第一候補です。"
        )
    elif shapiro_ok and not levene_ok:
        st.info(
            "両群とも正規性を棄却する十分な根拠はありませんが、等分散性は仮定しにくいため、"
            "この場合は Welch t-test が第一候補です。"
        )
    else:
        st.info(
            "正規性を仮定しにくい可能性があるため、"
            "この場合は Mann-Whitney U が第一候補です。"
        )

    st.caption(
        "※ p≥α は『差がないことの証明』ではなく、『その前提を棄却する十分な根拠が得られなかった』ことを意味します。"
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="検定結果をCSVでダウンロード",
        data=csv_bytes,
        file_name="test_results.csv",
        mime="text/csv",
    )
else:
    st.info("CSVをアップロードしてください。")
