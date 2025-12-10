import os
from datetime import datetime 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.font_manager as fm

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

# ---- 한글 폰트(NanumGothic) 설정 ----
# fonts/NanumGothic.ttf 위치는 프로젝트 구조에 맞게 필요하면 수정
font_path = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

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
    st.subheader("1. 히스토그램 - 월세 분포 분석")

    # bin 개수 슬라이더
    bins = st.slider("bin 개수 (구간 수)", min_value=10, max_value=60, value=30, step=5)

    # 한 줄에 3개의 히스토그램 (서울 전체 / 구 A / 구 B)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

    datasets = [
        ("서울 전체", seoul),
        (f"{gu_a}", df_a),
        (f"{gu_b}", df_b),
    ]

    # 🔹 서울+두 구 전체 월세 기준으로 x축 상한 결정 (99퍼센타일)
    all_rent = np.concatenate([
        seoul["월세금(만원)"].dropna().values,
        df_a["월세금(만원)"].dropna().values,
        df_b["월세금(만원)"].dropna().values,
    ])
    # 0원 이하 값 제거
    all_rent = all_rent[all_rent > 0]

    if len(all_rent) == 0:
        st.warning("월세 데이터가 없습니다.")
    else:
        x_max = np.percentile(all_rent, 99)   # 상위 1% 잘라내기

        for ax, (label, d) in zip(axes, datasets):
            # 결측치 제거 및 0원 이하 제거
            data = d["월세금(만원)"].dropna()
            data = data[data > 0]

            if len(data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "데이터 없음",
                    ha="center",
                    va="center",
                    fontproperties=font_prop,
                )
                ax.set_axis_off()
                continue

            # 🔹 서울 전체 기준 상위 1% 초과 값은 히스토그램에서 제외
            data = data[data <= x_max]

            # 🔹 각 지역별로 '비율(%)'이 되도록 정규화
            #    → 막대 높이 = (해당 구간 비중 * 100)
            weights = np.ones_like(data, dtype=float) / len(data) * 100

            ax.hist(
                data,
                bins=bins,
                range=(0, x_max),   # bin 경계를 0~x_max로 고정
                weights=weights,    # y축을 비율(%)로 만들기 위한 가중치
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_title(f"{label} (n={len(data)})", fontproperties=font_prop)
            ax.set_xlabel("월세 (만원)", fontproperties=font_prop)
            ax.set_ylabel("비율(%)", fontproperties=font_prop)
            ax.set_xlim(0, x_max)

            for tick in ax.get_xticklabels():
                tick.set_fontproperties(font_prop)
            for tick in ax.get_yticklabels():
                tick.set_fontproperties(font_prop)

        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "- 서울 전체와 두 개 구의 월세 분포를 **비율(%) 기준**으로 동시에 비교할 수 있습니다.\n"
            "- 서울 전체 기준 상위 1% 초과 고가 월세는 히스토그램에서 제외하고, 꼬리(극단값)로 따로 해석하면 됩니다.\n"
            "- 오른쪽 꼬리가 길수록 고가 월세가 일부 존재한다는 뜻으로 해석할 수 있습니다."
        )
# =====================================
# 2. BoxPlot – 신·중축 vs 구축 월세 비교
# =====================================
with tab_box:
    st.subheader("2. BoxPlot – 신·중축 vs 구축 ㎡당 월세 비교")

    # 0) 기본 컬럼 체크
    if "건축년도" not in df_filtered.columns:
        st.warning("데이터에 '건축년도' 컬럼이 없어 BoxPlot을 그릴 수 없습니다.")
    else:
        # 어떤 월세 컬럼을 쓸지 결정 (전용면적당 월세 우선, 없으면 월세금 사용)
        if "전용면적당 월세(만원/㎡)" in df_filtered.columns:
            rent_col = "전용면적당 월세(만원/㎡)"
            y_label = "전용면적당 월세 (만원/㎡)"
        elif "월세금(만원)" in df_filtered.columns:
            rent_col = "월세금(만원)"
            y_label = "월세 (만원)"
        else:
            st.warning("월세 관련 컬럼이 없어 BoxPlot을 그릴 수 없습니다.")
            st.stop()

        valid_years = df_filtered["건축년도"].dropna()
        if valid_years.empty:
            st.warning("건축년도 정보가 거의 없어 BoxPlot을 그릴 수 없습니다.")
        else:
            AGE_CUTOFF = 20
            current_year = datetime.now().year

            st.markdown(
                f"신·중축 vs 구축 기준: **건축 후 {AGE_CUTOFF}년 이하 → '신·중축', "
                f"{AGE_CUTOFF}년 초과 → '구축'** (기준연도: {current_year}년)"
            )

            def add_age_group(d: pd.DataFrame) -> pd.DataFrame:
                # 건축년도와 선택한 월세 컬럼 둘 다 있는 행만 사용
                d2 = d.dropna(subset=["건축년도", rent_col]).copy()
                d2["연식"] = current_year - d2["건축년도"]
                d2["연식그룹"] = np.where(
                    d2["연식"] <= AGE_CUTOFF,
                    "신·중축",
                    "구축",
                )
                return d2

            seoul_age = add_age_group(seoul)
            a_age = add_age_group(df_a)
            b_age = add_age_group(df_b)

            region_datasets_age = [
                ("서울 전체", seoul_age),
                (f"{gu_a}", a_age),
                (f"{gu_b}", b_age),
            ]

            # -------------------------------
            # 2-1) 연식 분포 히스토그램 (위)
            # -------------------------------
            fig_age, axes_age = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

            for ax, (label, d) in zip(axes_age, region_datasets_age):
                if d.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "데이터 부족",
                        ha="center",
                        va="center",
                        fontproperties=font_prop,
                    )
                    ax.set_axis_off()
                    continue

                counts = d["연식그룹"].value_counts()
                counts = counts.reindex(["신·중축", "구축"])
                ratios = counts / counts.sum() * 100

                ax.bar(ratios.index, ratios.values)
                ax.set_ylim(0, 100)
                ax.set_title(f"{label} (n={int(counts.sum())})", fontproperties=font_prop)
                ax.set_ylabel("비율(%)", fontproperties=font_prop)

                for tick in ax.get_xticklabels():
                    tick.set_fontproperties(font_prop)
                for tick in ax.get_yticklabels():
                    tick.set_fontproperties(font_prop)

            plt.tight_layout()
            st.pyplot(fig_age)

            # -------------------------------
            # 2-2) 신·중축 vs 구축 월세 BoxPlot (아래)
            # -------------------------------
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

            region_datasets = [
                ("서울 전체", seoul_age),
                (f"{gu_a}", a_age),
                (f"{gu_b}", b_age),
            ]

            for ax, (label, d) in zip(axes2, region_datasets):
                # 데이터가 없거나, 한 그룹만 있으면 표시 X
                if d.empty or d["연식그룹"].nunique() < 2:
                    ax.set_title(f"{label}\n데이터 부족", fontproperties=font_prop)
                    ax.axis("off")
                    continue

                d.boxplot(
                    column=rent_col,
                    by="연식그룹",
                    ax=ax,
                    grid=False,
                )
                ax.set_title(label, fontproperties=font_prop)
                ax.set_xlabel("", fontproperties=font_prop)
                ax.set_ylabel(y_label, fontproperties=font_prop)

                for tick in ax.get_xticklabels():
                    tick.set_fontproperties(font_prop)
                for tick in ax.get_yticklabels():
                    tick.set_fontproperties(font_prop)

            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig2)

            # 요약 통계표 (발표/해석용)
            summary_list = []
            for region_label, d in region_datasets:
                if d.empty or d["연식그룹"].nunique() == 0:
                    continue

                s = (
                    d.groupby("연식그룹")[rent_col]
                    .describe()[["count", "25%", "50%", "75%"]]
                    .rename(
                        columns={
                            "count": "표본수",
                            "25%": "1분위(Q1)",
                            "50%": "중앙값(Q2)",
                            "75%": "3분위(Q3)",
                        }
                    )
                    .reset_index()
                )
                s.insert(0, "지역", region_label)
                summary_list.append(s)

            if summary_list:
                st.write("##### 신·중축 vs 구축 ㎡당 월세 요약 통계")
                summary_df = pd.concat(summary_list, ignore_index=True)
                st.dataframe(summary_df)

            st.caption(
                "- 동일 면적 기준으로 **신·중축 vs 구축의 ㎡당 월세 수준과 변동성(IQR)**을 비교할 수 있습니다.\n"
                "- 신·중축의 중앙값이 구축보다 높으면, 같은 면적 대비 월세 부담이 더 크다는 뜻입니다.\n"
                "- 신·중축 상자의 폭(IQR)이 넓으면, 신축·중축 주택의 가격 분산이 크다는 의미로 해석할 수 있습니다."
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
                ax.set_title(f"{label}\n데이터 부족", fontproperties=font_prop)
                ax.axis("off")
                continue

            ax.scatter(d["보증금(만원)"], d["월세금(만원)"], alpha=0.4, s=10)
            ax.set_title(f"{label} (n={len(d)})", fontproperties=font_prop)
            ax.set_xlabel("보증금 (만원)", fontproperties=font_prop)
            ax.set_ylabel("월세 (만원)", fontproperties=font_prop)

            for tick in ax.get_xticklabels():
                tick.set_fontproperties(font_prop)
            for tick in ax.get_yticklabels():
                tick.set_fontproperties(font_prop)

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

    # 0) 단지명 / 건물명 컬럼 찾기
    building_col = None
    for col in ["단지명", "건물명"]:
        if col in seoul.columns:
            building_col = col
            break

    highlight_name = None
    idx_seoul, idx_a, idx_b = [], [], []

    # 1) 사용자에게 매물 이름 입력받기 (검색창)
    if building_col is not None:
        highlight_name = st.text_input(
            f"Q-Q Plot에서 확인하고 싶은 {building_col} 이름을 입력하세요 (부분일치 가능)"
        )

        def find_idx(df_in: pd.DataFrame):
            if not highlight_name:
                return []
            cand = df_in[
                df_in[building_col]
                .astype(str)
                .str.contains(highlight_name, case=False, na=False)
            ]
            return cand.index.tolist()

        if highlight_name:
            idx_seoul = find_idx(seoul)
            idx_a = find_idx(df_a)
            idx_b = find_idx(df_b)

            total = len(set(idx_seoul) | set(idx_a) | set(idx_b))
            if total == 0:
                st.warning(f"'{highlight_name}'을(를) 포함하는 계약을 찾지 못했습니다.")
            else:
                st.info(f"'{highlight_name}'을(를) 포함하는 계약 {total}건을 찾았습니다.")
    else:
        st.caption("※ 단지명/건물명 컬럼이 없어 개별 매물 표시 기능은 비활성화됩니다.")

    # 2) QQ Plot 함수: DataFrame + highlight index를 받아서 그림
    def qq_plot(ax, df_in: pd.DataFrame, label: str, highlight_idx=None):
        data = df_in["월세금(만원)"].dropna()
        if len(data) < 10:
            ax.set_title(f"{label}\n데이터 부족", fontproperties=font_prop)
            ax.axis("off")
            return

        # 정렬하면서 원래 index 유지
        sorted_data = data.sort_values()

        # probplot은 값만 넘기고, index는 따로 DataFrame으로 붙이기
        (osm, osr), (slope, intercept, r) = stats.probplot(
            sorted_data.values, dist="norm", fit=True
        )

        qq_df = pd.DataFrame(
            {"osm": osm, "osr": osr},
            index=sorted_data.index,  # ← 원래 행 index
        )

        # 전체 점
        ax.scatter(qq_df["osm"], qq_df["osr"], alpha=0.5, s=10, label="관측값")

        # 선택 매물 강조
        if highlight_idx:
            pts = qq_df.loc[qq_df.index.isin(highlight_idx)]
            if not pts.empty:
                ax.scatter(
                    pts["osm"],
                    pts["osr"],
                    s=80,
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=2,
                    label="선택 매물",
                )

                # 너무 많으면 복잡하니 앞 몇 개만 이름 라벨링
                if building_col is not None and building_col in df_in.columns:
                    for idx_row, row in pts.head(3).iterrows():
                        name = str(df_in.loc[idx_row, building_col])
                        ax.annotate(
                            name,
                            (row["osm"], row["osr"]),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=7,
                            fontproperties=font_prop,  # ← 한글 라벨
                        )

        # 참고선
        ax.plot(osm, slope * osm + intercept, color="red", linewidth=2, label="참고선")

        ax.set_title(
            f"{label} (n={len(data)}, R={r:.2f})",
            fontproperties=font_prop,
        )
        ax.set_xlabel("이론 분위수 (정규분포)", fontproperties=font_prop)
        ax.set_ylabel("관측 월세 (만원)", fontproperties=font_prop)
        ax.legend(loc="best", fontsize=8, prop=font_prop)

        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)

    # 3) 서브플롯 3개 그리기
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    qq_plot(axes4[0], seoul, "서울 전체", highlight_idx=idx_seoul)
    qq_plot(axes4[1], df_a, gu_a, highlight_idx=idx_a)
    qq_plot(axes4[2], df_b, gu_b, highlight_idx=idx_b)

    plt.tight_layout()
    st.pyplot(fig4)

    st.caption(
        "- 직선에서 크게 벗어난 점들이 **Outlier(극단값)**입니다.\n"
        "- 서울 전체와 각 구의 Q-Q Plot을 비교해 보면, 어떤 구에서 고가 월세 Outlier가 더 많이 나타나는지 설명할 수 있습니다.\n"
        "- 상단 입력창에 매물 이름을 입력하면, 해당 매물이 Q-Q Plot 상에서 어느 위치(극단값인지/평균 근처인지)에 있는지 확인할 수 있습니다."
    )
