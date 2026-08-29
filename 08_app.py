import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from textwrap import dedent

# ============================================================
# PAGE CONFIG — MUST BE FIRST
# ============================================================

st.set_page_config(
    page_title="Healthcare Desert AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL + DATA
# ============================================================

MODEL_PATH = "models/healthcare_desert_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
DATA_PATH = "data/Healthcare_Desert_Scored.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

df = pd.read_csv(DATA_PATH)

# ============================================================
# THEME
# ============================================================

NAVY = "#07182B"
NAVY_2 = "#0B2238"
NAVY_3 = "#102D47"
BUTTER = "#F6E7A1"
BUTTER_LIGHT = "#FFF4C4"
WHITE = "#F7F9FC"
MUTED = "#9EADBE"
BLUE = "#9ED8F5"
BLUE_2 = "#6FC3E8"
GREEN = "#9ED6B4"
ORANGE = "#F2C879"
RED = "#E99A9A"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    dedent(
        f"""
        <style>

        /* =====================================================
           GLOBAL
        ===================================================== */

        .stApp {{
            background:
                radial-gradient(
                    circle at 15% 0%,
                    rgba(246,231,161,0.10),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 95% 35%,
                    rgba(111,195,232,0.08),
                    transparent 28%
                ),
                linear-gradient(
                    135deg,
                    {NAVY} 0%,
                    #081D32 50%,
                    #0B263D 100%
                );

            color: {WHITE};
        }}

        .main .block-container {{
            max-width: 1380px;
            padding-top: 2.5rem;
            padding-bottom: 5rem;
        }}

        /* Remove default Streamlit decoration */

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(5,15,27,0.92);
        }}

        /* =====================================================
           SIDEBAR
        ===================================================== */

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #061426 0%,
                    #0A1F35 100%
                );

            border-right: 1px solid
            rgba(246,231,161,0.13);
        }}

        section[data-testid="stSidebar"] h2 {{
            font-family: Georgia, serif;
            color: {BUTTER};
            font-size: 22px;
        }}

        section[data-testid="stSidebar"] label {{
            color: {MUTED};
            font-weight: 600;
        }}

        /* =====================================================
           HERO
        ===================================================== */

        .hero {{
            padding: 20px 0 15px 0;
        }}

        .hero-title {{
    text-align: center;

    font-family: Georgia, "Times New Roman", serif;
    font-size: 58px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -1.5px;

    background:
        linear-gradient(
            90deg,
            #FFFFFF 0%,
            #FFF8D6 35%,
            {BUTTER} 100%
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    margin: 0 auto;
}}

        .hero-byline {{
            color: {BUTTER};
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 2px;
            margin-top: 12px;
            text-transform: uppercase;
        }}

        .hero-subtitle {{
            color: {MUTED};
            font-size: 17px;
            line-height: 1.65;
            max-width: 850px;
            margin-top: 14px;
        }}

        .hero-line {{
            height: 2px;
            margin-top: 28px;
            background:
                linear-gradient(
                    90deg,
                    {BUTTER},
                    rgba(246,231,161,0.25),
                    transparent
                );
        }}

        /* =====================================================
           SECTION HEADERS
        ===================================================== */

        h1, h2, h3 {{
            font-family: Georgia, serif !important;
        }}

        .section-title {{
            font-family: Georgia, serif;
            font-size: 31px;
            font-weight: 700;
            color: {WHITE};
            margin-top: 18px;
            margin-bottom: 5px;
        }}

        .section-description {{
            color: {MUTED};
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 18px;
        }}

        /* =====================================================
   SCORE AREA
   ===================================================== */

.score-panel {{
    background:
        linear-gradient(
            145deg,
            rgba(17,47,73,0.92),
            rgba(7,25,43,0.95)
        );

    border: 1px solid
    rgba(246,231,161,0.15);

    border-radius: 22px;
    padding: 10px 18px;

    box-shadow:
        0 18px 50px rgba(0,0,0,0.24);
}}

.insight-card {{
    background:
        linear-gradient(
            145deg,
            rgba(24,55,78,0.96),
            rgba(10,30,49,0.98)
        );

    border: 1px solid
    rgba(246,231,161,0.20);

    border-radius: 22px;
    padding: 32px;

    min-height: 365px;

    box-shadow:
        0 18px 45px rgba(0,0,0,0.25);
}}

.insight-label {{
    color: {BUTTER};
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}}

.insight-value {{
    color: {WHITE};
    font-size: 39px;
    font-weight: 800;
    margin-top: 9px;
    margin-bottom: 14px;
}}

.insight-text {{
    color: #DCE4ED;
    font-size: 16px;
    line-height: 1.7;
}}

.population-value {{
    color: {WHITE};
    font-size: 31px;
    font-weight: 800;
    margin-top: 6px;
}}

        /* =====================================================
   KPI CARDS
   ===================================================== */

.metric-card {{
    background:
        linear-gradient(
            145deg,
            rgba(17,48,76,0.96),
            rgba(9,29,49,0.96)
        );

    border: 1px solid
    rgba(158,216,245,0.13);

    border-radius: 17px;

    padding: 24px 22px;

    min-height: 130px;

    box-shadow:
        0 12px 32px rgba(0,0,0,0.20);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}}

.metric-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(246,231,161,0.45);
}}

.metric-label {{
    color: #B7C5D5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}}

.metric-value {{
    color: {WHITE};
    font-size: 34px;
    font-weight: 800;
    margin-top: 11px;
}}

        /* =====================================================
           CHART CONTAINERS
        ===================================================== */

        .chart-title {{
            font-family: Georgia, serif;
            color: {WHITE};
            font-size: 21px;
            font-weight: 700;
            margin: 10px 0 5px 0;
        }}

        .chart-note {{
            color: {MUTED};
            font-size: 12px;
            line-height: 1.5;
            margin-bottom: 10px;
        }}

        /* =====================================================
   PRIORITY CARDS
   ===================================================== */

.recommendation-card {{
    background:
        linear-gradient(
            145deg,
            rgba(17,43,66,0.98),
            rgba(8,27,45,0.98)
        );

    border: 1px solid
    rgba(255,255,255,0.08);

    border-left: 4px solid {BUTTER};

    border-radius: 16px;

    padding: 25px 28px;

    margin-bottom: 15px;

    box-shadow:
        0 12px 30px rgba(0,0,0,0.18);
}}

/* =====================================================
   PRIORITY CARD TITLES
   ===================================================== */

/* High priority = RED */
.recommendation-card.priority-high-card h3 {{
    color: #E99A9A;
    font-family: Georgia, serif;
    font-size: 25px;
    font-weight: 700;
    margin: 13px 0 13px 0;
}}

/* Medium priority = BUTTER YELLOW */
.recommendation-card.priority-medium-card h3 {{
    color: {BUTTER};
    font-family: Georgia, serif;
    font-size: 25px;
    font-weight: 700;
    margin: 13px 0 13px 0;
}}

/* =====================================================
   CURRENT VALUE + RECOMMENDED INTERVENTION
   ===================================================== */

.recommendation-card p {{
    color: #D3DCE6;
    font-size: 17px;
    line-height: 1.75;
}}

/* Current value itself = bigger + bold */
.recommendation-card .current-value {{
    color: {WHITE};
    font-size: 18px;
    font-weight: 800;
}}

/* Percentage = butter yellow */
.recommendation-card .current-value strong {{
    color: {BUTTER};
    font-size: 20px;
    font-weight: 800;
}}

/* =====================================================
   PRIORITY BADGES
   ===================================================== */

.priority-high,
.priority-medium {{
    font-size: 13px;
    padding: 6px 12px;
    letter-spacing: 1px;
}}

        /* =====================================================
   INFO CARDS
   ===================================================== */

.info-card {{
    background:
        rgba(15,43,67,0.75);

    border: 1px solid
    rgba(255,255,255,0.07);

    border-radius: 18px;

    padding: 28px;

    margin-top: 10px;
}}

.info-card p {{
    color: #D3DCE6;
    line-height: 1.75;
    font-size: 15px;
}}
        /* =====================================================
           WHAT THIS MEANS
           ===================================================== */

        .societal-label {{
            color: {BUTTER};
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 1.8px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }}

        .societal-main {{
            color: {WHITE};
            font-size: 18px;
            font-weight: 500;
            line-height: 1.75;
            margin: 0 0 20px 0;
        }}

        .societal-note {{
            color: {MUTED};
            font-size: 15px;
            line-height: 1.7;
            margin: 0;
        }}

        /* =====================================================
           DIVIDERS
        ===================================================== */

        hr {{
            border-color: rgba(246,231,161,0.10);
        }}

        /* =====================================================
           DATAFRAME
        ===================================================== */

        div[data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
        }}

        /* =====================================================
           BUTTONS
        ===================================================== */

        .stButton > button {{
            border-radius: 10px;
            border: 1px solid rgba(246,231,161,0.25);
            background: {NAVY_3};
            color: {WHITE};
            font-weight: 650;
        }}

        .stButton > button:hover {{
            border-color: {BUTTER};
            color: {BUTTER};
        }}

        </style>
        """
    ),
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">Healthcare Desert AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-line"></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'A data-driven dashboard for identifying healthcare '
    'accessibility gaps across Indian districts and highlighting '
    'areas requiring targeted intervention.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Select a district to estimate its **Healthcare Desert Score** "
    "and identify its level of healthcare access risk."
)

st.divider()
# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## District Selection")

states = sorted(
    df["State_UT"].dropna().unique()
)

selected_state = st.sidebar.selectbox(
    "Select State / UT",
    states
)

state_df = df[
    df["State_UT"] == selected_state
]

districts = sorted(
    state_df["District"].dropna().unique()
)

selected_district = st.sidebar.selectbox(
    "Select District",
    districts
)

# ============================================================
# GET DISTRICT DATA
# ============================================================

row = state_df[
    state_df["District"] == selected_district
].iloc[0]

X = row[feature_columns].to_frame().T

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    df[feature_columns].median()
)

prediction = model.predict(X)[0]

prediction = max(
    0,
    min(100, prediction)
)

# ============================================================
# RISK CATEGORY
# ============================================================

if prediction < 36.93:
    risk = "Low Risk"
    risk_color = GREEN
elif prediction < 63.52:
    risk = "Medium Risk"
    risk_color = ORANGE
else:
    risk = "High Risk"
    risk_color = RED

# ============================================================
# DISTRICT TITLE
# ============================================================

st.markdown(
    f"""
    <div class="section-title">
    {selected_district}, {selected_state}
    </div>

    <div class="section-description">
    District-level healthcare accessibility assessment
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SCORE DASHBOARD
# ============================================================

score_col, info_col = st.columns(
    [1.25, 1],
    gap="large"
)

# ============================================================
# GAUGE
# ============================================================

with score_col:

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={
                "suffix": " / 100",
                "font": {
                    "size": 42,
                    "color": WHITE
                }
            },
            title={
                "text": "Healthcare Desert Score",
                "font": {
                    "size": 19,
                    "color": BUTTER
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": MUTED
                },
                "bar": {
                    "color": BUTTER,
                    "thickness": 0.27
                },
                "bgcolor": NAVY_2,
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 36.93],
                        "color": "#183B56"
                    },
                    {
                         "range": [36.93, 63.52],
                         "color": "#315A75"
                    },
                    {
                        "range": [63.52, 100],
                        "color": "#665A35"
                    }
                ],
                "threshold": {
                    "line": {
                        "color": WHITE,
                        "width": 4
                    },
                    "thickness": 0.8,
                    "value": prediction
                }
            }
        )
    )

    gauge.update_layout(
        height=380,
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=15
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.markdown(
        '<div class="score-panel">',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        gauge,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================
# SCORE INFORMATION
# ============================================================

with info_col:

    rural_population = row[
        "Rural_Population_2011"
    ]

    st.markdown(
        dedent(
            f"""
            <div class="insight-card">

            <div class="insight-label">
            Current Assessment:- 
            </div>

            <div class="insight-value"
                 style="color:{risk_color};">
            {risk}
            </div>

            <div class="insight-text">
            The district has a predicted Healthcare Desert Score
            of <b>{prediction:.2f}</b> out of 100.
            </div>

            <br>

            <div class="insight-label">
            Score Interpretation
            </div>

            <div class="insight-text">
            0–36.93: Low Risk<br>
            36.93–63.52: Medium Risk<br>
            63.52–100: High Risk
            </div>

            <br>

            <div class="insight-label">
            Rural Population
            </div>

            <div class="population-value">
            {rural_population:,.0f}
            </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

# ============================================================
# DISTRICT VS NATIONAL BENCHMARK
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">District vs National Benchmark</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The selected district is compared against the median across all '
    'districts in the dataset. The national median is represented as 100%.'
    '</div>',
    unsafe_allow_html=True
)

benchmark_data = pd.DataFrame({
    "Indicator": [
        "PHCs / 100K",
        "CHCs / 100K",
        "Sub-Centres / 100K",
        "Health Insurance",
        "Women's Literacy",
        "Sanitation"
    ],
    "District": [
        row["PHCs_per_100k_rural"],
        row["CHCs_per_100k_rural"],
        row["SubCentres_per_100k_rural"],
        row["Health_Insurance_Coverage_pct"],
        row["Women_15_49_Literate_pct"],
        row["Improved_Sanitation_pct"]
    ],
    "National Median": [
        df["PHCs_per_100k_rural"].median(),
        df["CHCs_per_100k_rural"].median(),
        df["SubCentres_per_100k_rural"].median(),
        df["Health_Insurance_Coverage_pct"].median(),
        df["Women_15_49_Literate_pct"].median(),
        df["Improved_Sanitation_pct"].median()
    ]
}).dropna()

benchmark_chart = benchmark_data.copy()

for col in [
    "District",
    "National Median"
]:
    benchmark_chart[col] = (
        benchmark_chart[col]
        / benchmark_data["National Median"]
        * 100
    )

fig_benchmark = go.Figure()

fig_benchmark.add_trace(
    go.Bar(
        y=benchmark_chart["Indicator"],
        x=benchmark_chart["District"],
        name="Selected District",
        orientation="h",
        marker_color=BLUE,
        text=[
            f"{x:.0f}%"
            for x in benchmark_chart["District"]
        ],
        textposition="auto"
    )
)

fig_benchmark.add_trace(
    go.Scatter(
        y=benchmark_chart["Indicator"],
        x=[100] * len(benchmark_chart),
        mode="markers",
        name="National Median",
        marker=dict(
            color=BUTTER,
            size=12,
            symbol="diamond"
        )
    )
)

fig_benchmark.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=WHITE),
    margin=dict(l=10, r=20, t=15, b=45),
    xaxis=dict(
        title="Index (National Median = 100)",
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False
    ),
    yaxis=dict(
        title="",
        gridcolor="rgba(255,255,255,0.03)"
    ),
    legend=dict(
        orientation="h",
        y=1.08,
        x=0
    ),
    bargap=0.28
)

st.plotly_chart(
    fig_benchmark,
    width="stretch",
    config={"displayModeBar": False}
)

st.caption(
    "A value of 100 represents the national district-level median. "
    "Values below 100 indicate that the selected district is below the benchmark."
)

# ============================================================
# HEALTHCARE INFRASTRUCTURE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Healthcare Infrastructure</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Healthcare facilities available within the selected district.'
    '</div>',
    unsafe_allow_html=True
)

infra_data = [
    (
        "Primary Health Centres",
        row["PHCs"]
    ),
    (
        "Community Health Centres",
        row["CHCs"]
    ),
    (
        "Sub-Centres",
        row["Sub_Centres"]
    ),
    (
        "Sub-Divisional Hospitals",
        row["Sub_Divisional_Hospitals"]
    ),
    (
        "District Hospitals",
        row["District_Hospitals"]
    )
]

infra_cols = st.columns(5)

for col, (label, value) in zip(
    infra_cols,
    infra_data
):

    with col:

        st.markdown(
            dedent(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                {label}
                </div>

                <div class="metric-value">
                {value:,.0f}
                </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

infra_df = pd.DataFrame(
    infra_data,
    columns=[
        "Facility",
        "Number"
    ]
)

infra_df = infra_df.sort_values(
    "Number",
    ascending=True
)

fig_infra = px.bar(
    infra_df,
    x="Number",
    y="Facility",
    orientation="h",
    text="Number"
)

fig_infra.update_traces(
    marker_color=BLUE,
    textposition="outside",
    textfont_color=WHITE
)

fig_infra.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=WHITE),
    margin=dict(l=10, r=60, t=20, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.02)"
    ),
    showlegend=False
)

st.plotly_chart(
    fig_infra,
    width="stretch",
    config={"displayModeBar": False}
)

# ============================================================
# HEALTHCARE AVAILABILITY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Healthcare Availability</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Healthcare facility availability relative to the district rural population.'
    '</div>',
    unsafe_allow_html=True
)

availability_data = pd.DataFrame({
    "Indicator": [
        "PHCs / 100K",
        "CHCs / 100K",
        "Sub-Centres / 100K"
    ],
    "District": [
        row["PHCs_per_100k_rural"],
        row["CHCs_per_100k_rural"],
        row["SubCentres_per_100k_rural"]
    ],
    "National Median": [
        df["PHCs_per_100k_rural"].median(),
        df["CHCs_per_100k_rural"].median(),
        df["SubCentres_per_100k_rural"].median()
    ]
}).dropna()

availability_cols = st.columns(3)

for col, (_, item) in zip(
    availability_cols,
    availability_data.iterrows()
):

    value = item["District"]

    with col:

        st.markdown(
            dedent(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                {item["Indicator"]}
                </div>

                <div class="metric-value">
                {value:.2f}
                </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

fig_availability = go.Figure()

fig_availability.add_trace(
    go.Bar(
        y=availability_data["Indicator"],
        x=availability_data["District"],
        name="Selected District",
        orientation="h",
        marker_color=BLUE,
        text=[
            f"{x:.2f}"
            for x in availability_data["District"]
        ],
        textposition="auto"
    )
)

fig_availability.add_trace(
    go.Scatter(
        y=availability_data["Indicator"],
        x=availability_data["National Median"],
        mode="markers",
        name="National Median",
        marker=dict(
            color=BUTTER,
            size=13,
            symbol="diamond"
        )
    )
)

fig_availability.update_layout(
    height=350,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=WHITE),
    margin=dict(l=10, r=40, t=20, b=35),
    xaxis=dict(
        title="Facilities per 100K rural population",
        gridcolor="rgba(255,255,255,0.07)"
    ),
    yaxis=dict(
        title=""
    ),
    legend=dict(
        orientation="h",
        y=1.10,
        x=0
    ),
    bargap=0.30
)

st.plotly_chart(
    fig_availability,
    width="stretch",
    config={"displayModeBar": False}
)

# ============================================================
# SOCIOECONOMIC INDICATORS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Socioeconomic Indicators</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Socioeconomic conditions that influence healthcare access and wellbeing.'
    '</div>',
    unsafe_allow_html=True
)

socioeconomic_data = pd.DataFrame({
    "Indicator": [
        "Electricity",
        "Drinking Water",
        "Sanitation",
        "Clean Cooking Fuel",
        "Health Insurance",
        "Women's Literacy"
    ],
    "Percentage": [
        row["Households_Electricity_pct"],
        row["Improved_Drinking_Water_pct"],
        row["Improved_Sanitation_pct"],
        row["Clean_Cooking_Fuel_pct"],
        row["Health_Insurance_Coverage_pct"],
        row["Women_15_49_Literate_pct"]
    ]
}).dropna()

fig_socio = px.bar(
    socioeconomic_data,
    x="Percentage",
    y="Indicator",
    orientation="h",
    text="Percentage"
)

fig_socio.update_traces(
    marker_color=BUTTER,
    texttemplate="%{text:.1f}%",
    textposition="outside",
    textfont_color=WHITE
)

fig_socio.update_layout(
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=WHITE),
    margin=dict(l=10, r=70, t=20, b=40),
    xaxis=dict(
        title="Percentage of households / population",
        range=[
            0,
            min(
                110,
                max(
                    socioeconomic_data["Percentage"].max() * 1.15,
                    100
                )
            )
        ],
        gridcolor="rgba(255,255,255,0.07)"
    ),
    yaxis=dict(
        title=""
    ),
    showlegend=False
)

st.plotly_chart(
    fig_socio,
    width="stretch",
    config={"displayModeBar": False}
)

with st.expander(
    "View detailed socioeconomic values"
):

    display_data = socioeconomic_data.copy()

    display_data["Percentage"] = (
        display_data["Percentage"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        display_data,
        width="stretch",
        hide_index=True
    )

# ============================================================
# PRIORITY AREAS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Priority Areas for Improvement</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Potential intervention areas are identified using district-level '
    'indicator thresholds and healthcare availability measures.'
    '</div>',
    unsafe_allow_html=True
)

recommendations = []

# ============================================================
# HEALTH INSURANCE
# ============================================================

insurance = row[
    "Health_Insurance_Coverage_pct"
]

if insurance < 25:

    recommendations.append({
        "Area": "Health Insurance Coverage",
        "Current Value": f"{insurance:.2f}%",
        "Priority": "High",
        "Recommendation":
            "Increase awareness and enrolment in government "
            "health insurance schemes, especially among rural households."
    })

elif insurance < 50:

    recommendations.append({
        "Area": "Health Insurance Coverage",
        "Current Value": f"{insurance:.2f}%",
        "Priority": "Medium",
        "Recommendation":
            "Improve awareness and accessibility of health "
            "insurance schemes."
    })

# ============================================================
# SANITATION
# ============================================================

sanitation = row[
    "Improved_Sanitation_pct"
]

if sanitation < 50:

    recommendations.append({
        "Area": "Sanitation",
        "Current Value": f"{sanitation:.2f}%",
        "Priority": "High",
        "Recommendation":
            "Improve access to household sanitation facilities "
            "and strengthen sanitation awareness programmes."
    })

elif sanitation < 70:

    recommendations.append({
        "Area": "Sanitation",
        "Current Value": f"{sanitation:.2f}%",
        "Priority": "Medium",
        "Recommendation":
            "Expand sanitation infrastructure and encourage "
            "consistent usage of sanitation facilities."
    })

# ============================================================
# CLEAN COOKING
# ============================================================

cooking = row[
    "Clean_Cooking_Fuel_pct"
]

if cooking < 50:

    recommendations.append({
        "Area": "Clean Cooking Fuel",
        "Current Value": f"{cooking:.2f}%",
        "Priority": "High",
        "Recommendation":
            "Improve access to affordable clean cooking fuels "
            "and increase awareness of cleaner alternatives."
    })

elif cooking < 70:

    recommendations.append({
        "Area": "Clean Cooking Fuel",
        "Current Value": f"{cooking:.2f}%",
        "Priority": "Medium",
        "Recommendation":
            "Increase adoption of cleaner and affordable "
            "household cooking solutions."
    })

# ============================================================
# WOMEN'S LITERACY
# ============================================================

literacy = row[
    "Women_15_49_Literate_pct"
]

if literacy < 50:

    recommendations.append({
        "Area": "Women's Literacy",
        "Current Value": f"{literacy:.2f}%",
        "Priority": "High",
        "Recommendation":
            "Strengthen adult education, women's literacy "
            "programmes and community-level awareness initiatives."
    })

elif literacy < 70:

    recommendations.append({
        "Area": "Women's Literacy",
        "Current Value": f"{literacy:.2f}%",
        "Priority": "Medium",
        "Recommendation":
            "Expand women's education and adult literacy "
            "programmes to improve long-term health outcomes."
    })

# ============================================================
# MATERNAL HEALTH
# ============================================================

anc = row[
    "ANC_4plus_pct"
]

if anc < 50:

    recommendations.append({
        "Area": "Maternal Healthcare",
        "Current Value": f"{anc:.2f}%",
        "Priority": "High",
        "Recommendation":
            "Improve access to antenatal care through stronger "
            "outreach, health-worker visits and referral systems."
    })

elif anc < 70:

    recommendations.append({
        "Area": "Maternal Healthcare",
        "Current Value": f"{anc:.2f}%",
        "Priority": "Medium",
        "Recommendation":
            "Increase awareness and accessibility of regular "
            "antenatal care services."
    })

# ============================================================
# CHILD HEALTH
# ============================================================

stunting = row[
    "Under5_Stunted_pct"
]

if stunting > 35:

    recommendations.append({
        "Area": "Child Nutrition",
        "Current Value": f"{stunting:.2f}%",
        "Priority": "High",
        "Recommendation":
            "Strengthen nutrition programmes, early screening "
            "and access to child healthcare services."
    })

# ============================================================
# DISPLAY PRIORITIES
# ============================================================

if recommendations:

    recommendations_df = pd.DataFrame(
        recommendations
    )

    priority_order = {
        "High": 1,
        "Medium": 2
    }

    recommendations_df["Sort"] = (
        recommendations_df["Priority"]
        .map(priority_order)
    )

    recommendations_df = (
        recommendations_df
        .sort_values("Sort")
        .drop(columns=["Sort"])
    )

    high_count = (
        recommendations_df["Priority"] == "High"
    ).sum()

    medium_count = (
        recommendations_df["Priority"] == "Medium"
    ).sum()

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    summary_cols = st.columns(3)

    summary_values = [
        ("Priority Areas", len(recommendations_df)),
        ("High Priority", high_count),
        ("Medium Priority", medium_count)
    ]

    for col, (label, value) in zip(
        summary_cols,
        summary_values
    ):

        with col:

            st.markdown(
                dedent(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

    # ========================================================
    # RECOMMENDED INTERVENTIONS
    # ========================================================

    st.markdown(
        "### Recommended Interventions"
    )

    for _, rec in recommendations_df.iterrows():

        if rec["Priority"] == "High":

            priority_class = "priority-high-card"

            badge = (
                '<span class="priority-high">'
                'HIGH PRIORITY'
                '</span>'
            )

        else:

            priority_class = "priority-medium-card"

            badge = (
                '<span class="priority-medium">'
                'MEDIUM PRIORITY'
                '</span>'
            )

        st.markdown(
            dedent(
                f"""
                <div class="recommendation-card {priority_class}">
                    {badge}
                    <h3>{rec["Area"]}</h3>
                    <p class="current-value">
                        <b>Current value:</b>
                        <strong>{rec["Current Value"]}</strong>
                    </p>
                    <p>
                        <b>Recommended intervention:</b>
                        {rec["Recommendation"]}
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True
        )

else:

    st.success(
        "No major improvement gaps were identified "
        "based on the selected indicators."
    )
# ============================================================
# WHAT THIS MEANS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">What This Means</div>',
    unsafe_allow_html=True
)

if prediction >= 63.52:

    societal_text = (
        "This district shows a high level of healthcare access risk. "
        "Targeted investment in healthcare infrastructure, coverage "
        "and social development indicators should be prioritised."
    )

elif prediction >= 36.93:

    societal_text = (
        "This district shows a moderate level of healthcare access risk. "
        "Focused improvements in the identified priority areas could "
        "help reduce healthcare accessibility gaps."
    )

else:
    societal_text = (
        "This district shows relatively lower healthcare access risk. "
        "Existing healthcare and social indicators should be maintained "
        "while continuing to address local gaps."
    )

st.markdown(
    f"""<div class="info-card">
<div class="societal-label">Societal Interpretation</div>
<div class="societal-main">{societal_text}</div>
<div class="societal-note">The model is intended to support identification, prioritisation and further investigation of healthcare accessibility gaps.</div>
</div>""",
    unsafe_allow_html=True
)

# ============================================================
# LIMITATIONS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Limitations</div>',
    unsafe_allow_html=True
)

st.markdown(
    dedent(
        f"""
        <div class="info-card">

        <p>
        <b> ~ Estimated score:</b>
        The Healthcare Desert Score is an estimated score based
        on the district-level indicators available in the dataset.
        </p>

        <p>
        <b> ~ Model interpretation:</b>
        The model identifies patterns and relationships in the data
        but does not establish direct cause-and-effect relationships.
        </p>

        <p>
        <b> ~ Data dependency:</b>
        The analysis depends on the quality, completeness and
        time period of the underlying data.
        </p>

        <p>
        <b> ~ Planning use:</b>
        The recommendations should support prioritisation and
        further ground-level assessment rather than replace
        local evaluation.
        </p>

        </div>
        """
    ),
    unsafe_allow_html=True
)

# ============================================================
# CONCLUSION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Conclusion</div>',
    unsafe_allow_html=True
)

st.markdown(
    dedent(
        f"""
        <div class="info-card">

        <p style="font-size:16px;">
        The <b> ~ Healthcare Desert AI</b> provides a data-driven
        approach to identifying healthcare accessibility gaps
        across Indian districts.
        </p>

        <p>
         ~ By combining healthcare infrastructure, facility availability
        and socioeconomic indicators, the model estimates district-level
        healthcare access risk and highlights potential areas requiring
        attention.
        </p>

        <p>
         ~ The identified priority areas and recommended interventions
        can support more targeted healthcare planning, resource
        allocation and future investigation.
        </p>

        </div>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Model Information</div>',
    unsafe_allow_html=True
)

with st.expander(
    "View methodology and model performance"
):

    st.markdown(
        dedent(
            f"""
            <div class="info-card">

            <p>
            The Healthcare Desert AI model uses a
            <b>Random Forest Regression</b> algorithm to estimate
            the Healthcare Desert Score.
            </p>

            <p>
            The model was trained using district-level socioeconomic,
            population, maternal health, child health and healthcare
            infrastructure indicators.
            </p>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    performance = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "R² Score"
        ],
        "Value": [
            11.75,
            15.18,
            0.6513
        ]
    })

    st.dataframe(
        performance,
        width="stretch",
        hide_index=True
    )


st.markdown(
    dedent(
        f"""
        <div style="
            text-align:center;
            color:{MUTED};
            font-size:12px;
            margin-top:45px;
            padding-top:20px;
            border-top:1px solid rgba(246,231,161,0.10);
        ">
        Healthcare Desert AI · District-level healthcare accessibility analysis
        <br>
        AI Model by Shravani Jadhav
        </div>
        """
    ),
    unsafe_allow_html=True
)
