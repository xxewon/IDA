import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(page_title="서울 전월세 4플롯 분석", layout="wide")

# 한글 폰트 (윈도우 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

st.title("서울 전월세 분석 – 4가지 플롯 전용 앱")

st.markdown(
    """
    이 앱은 **4가지 플롯만** 제공합니다.

    1. 히스토그램 – 월세 분포 분석  
    2. BoxPlot – 신축 vs 노후주택 월세 비교  
    3. Scatter Plot – 보증금 대비 월세 수준 비교 (서울 vs 선택 구 2개)  
    4. Q-Q Plot – Outlier와 정규성, 서울 vs 선택 구 2개 비교  

    좌측에서 주택유형을 선택하고, 각 탭에서 비교할 구를 고르면 됩니다.
    """
)

# -------------------------
# 데이터 로딩
# -------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    서울 아파트/오피스텔/연립다세대 월세 데이터를 모두 불러와 하나의 DataFrame으로 합친다.
    파일 경로는 실제 파일명에 맞게 수정해서 사용하세요.
    """
    file_paths = {
        "아파트": "APT_역거리_지수감쇠_가격추가.csv",   # ✅ 실제 파일명에 맞게 수정
        "오피스텔": "OPI_역거리_지수감쇠_가격추가.csv",
        "연립다세대": "DSD_역거리_지수감쇠_가격추가.csv",
    }

    dfs = []
    for htype, path in file_paths.items():
        if not os.path.exists(path):
            st.warning(f"{htype} 데이터 파일을 찾을 수 없습니다: {path}")
            continue

        tmp = pd.read_csv(path, encoding="utf-8-sig")

        # 주택유형 컬럼
        if "주택유형" not in tmp.columns:
            tmp["주택유형"] = htype

        # 시군구 → 구 추출 (예: '서울특별시 관악구 봉천동')
        if "구" not in tmp.columns and "시군구" in tmp.columns:
            parts = tmp["시군구"].astype(str).str.split()
            tmp["구"] = parts.str[1]

        # 월세/보증금/건축년도 숫자 처리
        for col in ["보증금(만원)", "월세금(만원)", "건축년도"]:
            if col in tmp.columns:
                tmp[col] = (
                    tmp[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                )
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        # 서울 데이터만 사용 (시군구가 있으면 서울만 필터)
        if "시군구" in tmp.columns:
            tmp = tmp[tmp["시군구"].astype(str).str.contains("서울")]

        # 월세 > 0 인 거래만 사용
        if "월세금(만원)" in tmp.columns:
            tmp = tmp[tmp["월세금(만원)"] > 0]

        dfs.append(tmp)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # 사용에 필요한 컬럼만 남기기 (있을 때만)
    keep_cols = [
        "주택유형",
        "시군구",
        "구",
        "보증금(만원)",
        "월세금(만원)",
        "건축년도",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


df = load_data()

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 상단 load_data()의 파일 경로를 확인하세요.")
    st.stop()

# -------------------------
# 공통 설정 (사이드바)
# -------------------------
st.sidebar.header("공통 설정")

housing_types = ["전체"] + sorted(df["주택유형"].dropna().unique().tolist())
selected_housing = st.sidebar.selectbox("주택유형 선택", housing_types, index=0)

# 주택유형 필터
if selected_housing != "전체":
    df_filtered = df[df["주택유형"] == selected_housing].copy()
else:
    df_filtered = df.copy()

# 구 목록
all_gu = sorted(df_filtered["구"].dropna().unique().tolist())

if len(all_gu) < 2:
    st.error("구 정보가 충분하지 않습니다. 데이터에 '구' 컬럼이 있는지 확인하세요.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write("각 탭에서 사용할 **구A, 구B**를 선택하세요.")

default_gu1 = all_gu[0]
default_gu2 = all_gu[1] if len(all_gu) > 1 else all_gu[0]

gu_a = st.sidebar.selectbox("구A 선택", all_gu, index=all_gu.index(default_gu1))
gu_b = st.sidebar.selectbox("구B 선택", all_gu, index=all_gu.index(default_gu2))

if gu_a == gu_b:
    st.sidebar.warning("구A와 구B가 같으면 비교가 어려우니 가능하면 다른 구를 선택하세요.")

# 편의용 서브셋
seoul = df_filtered.copy()
df_a = df_filtered[df_filtered["구"] == gu_a].copy()
df_b = df_filtered[df_filtered["구"] == gu_b].copy()

# -------------------------
# 탭 구성 (4개 분석 기능만)
# -------------------------
tab_hist, tab_box, tab_scatter, tab_qq = st.tabs(
    ["1. 히스토그램", "2. BoxPlot (신축 vs 노후)", "3. Scatter Plot", "4. Q-Q Plot"]
)

# =====================================
# 1. 히스토그램 – 월세 분포 분석
# =====================================
with tab_hist:
    st.subheader("1. 히스토그램 – 월세 분포 분석")

    bins = st.slider("bin 개수 (구간 수)", min_value=10, max_value=60, value=30, step=5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

    datasets = [
        ("서울 전체", seoul),
        (f"{gu_a}", df_a),
        (f"{gu_b}", df_b),
    ]

    for ax, (label, d) in zip(axes, datasets):
        data = d["월세금(만원)"].dropna()
        ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")
        ax.set_title(f"{label} (n={len(data)})")
        ax.set_xlabel("월세 (만원)")
        ax.set_ylabel("거래 건수")

    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "- 서울 전체와 두 개 구의 월세 분포를 동시에 비교할 수 있습니다.\n"
        "- 오른쪽 꼬리가 길수록 고가 월세가 일부 존재한다는 뜻입니다."
    )

# =====================================
# 2. BoxPlot – 신축 vs 노후주택 월세 비교
# =====================================
with tab_box:
    st.subheader("2. BoxPlot – 신축 vs 노후주택 월세 비교")

    if "건축년도" not in df_filtered.columns:
        st.warning("데이터에 '건축년도' 컬럼이 없어 BoxPlot을 그릴 수 없습니다.")
    else:
        valid_years = df_filtered["건축년도"].dropna()
        if valid_years.empty:
            st.warning("건축년도 정보가 거의 없어 BoxPlot을 그릴 수 없습니다.")
        else:
            year_min = int(valid_years.min())
            year_max = int(valid_years.max())

            new_cut = st.slider(
                "신축 기준 건축년도 (이 해 이상이면 '신축', 미만이면 '노후주택')",
                min_value=year_min,
                max_value=year_max,
                value=min(2010, year_max),
            )

            def add_age_group(d: pd.DataFrame) -> pd.DataFrame:
                d = d.dropna(subset=["건축년도", "월세금(만원)"]).copy()
                d["연식그룹"] = np.where(d["건축년도"] >= new_cut, "신축", "노후주택")
                return d

            seoul_age = add_age_group(seoul)
            a_age = add_age_group(df_a)
            b_age = add_age_group(df_b)

            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

            for ax, (label, d) in zip(
                axes2,
                [("서울 전체", seoul_age), (f"{gu_a}", a_age), (f"{gu_b}", b_age)],
            ):
                if d.empty or d["연식그룹"].nunique() < 2:
                    ax.set_title(f"{label}\n데이터 부족")
                    ax.axis("off")
                    continue

                d.boxplot(
                    column="월세금(만원)",
                    by="연식그룹",
                    ax=ax,
                    grid=False,
                )
                ax.set_title(label)
                ax.set_xlabel("")
                ax.set_ylabel("월세 (만원)")

            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig2)

            st.caption(
                "- 각 영역에서 **신축 vs 노후주택의 월세 수준과 변동성(IQR)**을 비교할 수 있습니다.\n"
                "- 신축의 중앙값이 높고 상자 폭이 크면, 신축 프리미엄과 함께 가격 분산도 크다는 뜻입니다."
            )

# =====================================
# 3. Scatter Plot – 보증금 vs 월세
# =====================================
with tab_scatter:
    st.subheader("3. Scatter Plot – 보증금 vs 월세 (서울 vs 구A vs 구B)")

    needed_cols = {"보증금(만원)", "월세금(만원)"}
    if not needed_cols.issubset(df_filtered.columns):
        st.warning("데이터에 '보증금(만원)' 또는 '월세금(만원)' 컬럼이 없어 산점도를 그릴 수 없습니다.")
    else:
        max_points = st.slider(
            "표시할 최대 점 개수 (무작위 샘플링)", min_value=200, max_value=5000, value=2000, step=200
        )

        def prep_scatter(d: pd.DataFrame) -> pd.DataFrame:
            d = d.dropna(subset=["보증금(만원)", "월세금(만원)"]).copy()
            d = d[(d["보증금(만원)"] > 0) & (d["월세금(만원)"] > 0)]
            if len(d) > max_points:
                d = d.sample(max_points, random_state=42)
            return d

        seoul_s = prep_scatter(seoul)
        a_s = prep_scatter(df_a)
        b_s = prep_scatter(df_b)

        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for ax, (label, d) in zip(
            axes3,
            [("서울 전체", seoul_s), (f"{gu_a}", a_s), (f"{gu_b}", b_s)],
        ):
            if d.empty:
                ax.set_title(f"{label}\n데이터 부족")
                ax.axis("off")
                continue

            ax.scatter(d["보증금(만원)"], d["월세금(만원)"], alpha=0.4, s=10)
            ax.set_title(f"{label} (n={len(d)})")
            ax.set_xlabel("보증금 (만원)")
            ax.set_ylabel("월세 (만원)")

        plt.tight_layout()
        st.pyplot(fig3)

        st.caption(
            "- 같은 보증금 수준에서 점들이 더 **위쪽에 몰린 구**는 `보증금 대비 월세 부담이 큰 구`로 해석할 수 있습니다.\n"
            "- 반대로 같은 보증금에서 월세가 상대적으로 낮으면 `보증금 위주 계약이 많은 구`로 이야기할 수 있습니다."
        )

# =====================================
# 4. Q-Q Plot – 서울 vs 구A vs 구B
# =====================================
with tab_qq:
    st.subheader("4. Q-Q Plot – 정규성 & Outlier (서울 vs 구A vs 구B)")

    def qq_plot(ax, data: pd.Series, label: str):
        data = data.dropna()
        if len(data) < 10:
            ax.set_title(f"{label}\n데이터 부족")
            ax.axis("off")
            return

        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", fit=True)
        ax.scatter(osm, osr, alpha=0.5, s=10, label="관측값")
        ax.plot(osm, slope * osm + intercept, color="red", linewidth=2, label="참고선")
        ax.set_title(f"{label} (n={len(data)}, R={r:.2f})")
        ax.set_xlabel("이론 분위수 (정규분포)")
        ax.set_ylabel("관측 월세 (만원)")
        ax.legend(loc="best", fontsize=8)

    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    qq_plot(axes4[0], seoul["월세금(만원)"], "서울 전체")
    qq_plot(axes4[1], df_a["월세금(만원)"], gu_a)
    qq_plot(axes4[2], df_b["월세금(만원)"], gu_b)

    plt.tight_layout()
    st.pyplot(fig4)

    st.caption(
        "- 직선에서 크게 벗어난 점들이 **Outlier(극단값)**입니다.\n"
        "- 서울 전체와 각 구의 Q-Q Plot을 비교해 보면, 어떤 구에서 고가 월세 Outlier가 더 많이 나타나는지 설명할 수 있습니다."
    )
