import streamlit as st
import pandas as pd
import os
import requests
import datetime
import io
import plotly.graph_objects as go
import numpy as np

# 1. 전체 앱에 나눔고딕 폰트 적용 (Google Fonts 웹폰트 사용)
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Nanum Gothic', 'NanumGothic', 'Malgun Gothic', Arial, sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 대제목
st.markdown(
    "<span style='font-size:18pt; font-weight:bold; color:#3399FF;'>◎ 종목검색</span>",
    unsafe_allow_html=True
)

# 3. 엑셀 데이터 로딩
EXCEL_PATH = 'kospi.xlsx'
if not os.path.exists(EXCEL_PATH):
    st.error(f"엑셀 파일이 존재하지 않습니다: {EXCEL_PATH}")
    st.stop()
df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip()
if '종목명' not in df.columns:
    st.error("'종목명' 컬럼이 엑셀 파일에 없습니다.")
    st.stop()

# 4. 종목 선택 (초기값 없음, placeholder)
stock_names = df['종목명'].dropna().unique().tolist()
selected_name = st.selectbox(
    "KOSPI 종목",
    options=stock_names,
    index=None,
    placeholder="종목을 검색 또는 입력하세요"
)

if selected_name:
    row = df[df['종목명'] == selected_name].iloc[0]
    code = str(row['종목코드']).zfill(6) if '종목코드' in df.columns else '-'
    market = row['시장구분'] if '시장구분' in df.columns else '-'
    industry = row['업종명'] if '업종명' in df.columns else '-'

    st.markdown(
        f"<b>선택종목명</b>: {selected_name} &nbsp; | &nbsp; "
        f"<b>종목코드</b>: {code} &nbsp; | &nbsp; "
        f"<b>시장구분</b>: {market} &nbsp; | &nbsp; "
        f"<b>업종명</b>: {industry}",
        unsafe_allow_html=True
    )

    # 날짜 입력
    today = datetime.date.today()
    default_start = datetime.date(2025, 1, 1)
    default_end = today
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작 날짜", default_start, key="start")
    with col2:
        end_date = st.date_input("종료 날짜", default_end, key="end")

    if start_date > end_date:
        st.error("시작날짜가 종료날짜보다 느릴수 없습니다.")

    # 검색단위
    timeframe_options = ["Day", "Week", "Month"]
    timeframe = st.radio(
        "검색 단위 선택",
        options=timeframe_options,
        index=1,
        horizontal=True
    )

    # 조회 버튼
    if st.button("조회"):
        st.session_state['result'] = {
            "selected_name": selected_name,
            "code": code,
            "market": market,
            "industry": industry,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe
        }
        # 데이터 요청
        timeframe_map = {"Day": "day", "Week": "week", "Month": "month"}
        timeframe_api = timeframe_map[timeframe]
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        symbol = code

        url = (
            "https://m.stock.naver.com/front-api/external/chart/domestic/info"
            f"?symbol={symbol}&requestType=1&startTime={start_str}&endTime={end_str}&timeframe={timeframe_api}"
        )

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=5)
            text = res.text.strip()
            df_price = pd.read_csv(io.StringIO(text))
            df_price = df_price.loc[:, ~df_price.columns.str.contains('^Unnamed')]
            clean_columns = []
            for col in df_price.columns:
                col_clean = col.replace("'", "").replace("[", "").replace("]", "").replace('"', "").strip()
                clean_columns.append(col_clean)
            df_price.columns = clean_columns
            df_price = df_price.rename(columns={
                'date': '날짜',
                'open': '시가',
                'close': '종가',
                'high': '고가',
                'low': '저가',
                'volume': '거래량',
                'frgn_rate': '외국인소진율'
            })
            df_price = df_price.replace({np.nan: None}).dropna()
            if '날짜' in df_price.columns:
                df_price['날짜'] = df_price['날짜'].astype(str).str.replace(r'[\[\]"]', '', regex=True)
                df_price['날짜'] = df_price['날짜'].apply(
                    lambda x: datetime.datetime.strptime(x, "%Y%m%d").strftime("%Y-%m-%d")
                    if x.isdigit() and len(x) == 8 else x
                )

            st.session_state['df_price'] = df_price

            # 1년간 변동추이 데이터도 저장
            one_year_ago = today - datetime.timedelta(days=365)
            start_1y = one_year_ago.strftime('%Y%m%d')
            end_1y = today.strftime('%Y%m%d')
            url_1y = (
                "https://m.stock.naver.com/front-api/external/chart/domestic/info"
                f"?symbol={symbol}&requestType=1&startTime={start_1y}&endTime={end_1y}&timeframe=week"
            )
            res_1y = requests.get(url_1y, headers=headers, timeout=5)
            text_1y = res_1y.text.strip()
            df_1y = pd.read_csv(io.StringIO(text_1y))
            df_1y = df_1y.loc[:, ~df_1y.columns.str.contains('^Unnamed')]
            clean_columns = []
            for col in df_1y.columns:
                col_clean = col.replace("'", "").replace("[", "").replace("]", "").replace('"', "").strip()
                clean_columns.append(col_clean)
            df_1y.columns = clean_columns
            df_1y = df_1y.rename(columns={
                'date': '날짜',
                'open': '시가',
                'close': '종가',
                'high': '고가',
                'low': '저가',
                'volume': '거래량',
                'frgn_rate': '외국인소진율'
            })
            df_1y = df_1y.replace({np.nan: None}).dropna()
            if '날짜' in df_1y.columns:
                df_1y['날짜'] = df_1y['날짜'].astype(str).str.replace(r'[\[\]"]', '', regex=True)
                df_1y['날짜'] = df_1y['날짜'].apply(
                    lambda x: datetime.datetime.strptime(x, "%Y%m%d").strftime("%Y-%m-%d")
                    if x.isdigit() and len(x) == 8 else x
                )
            st.session_state['df_1y'] = df_1y

        except Exception as e:
            st.error(f"데이터 요청/파싱 오류: {e}")

# ------------------- 결과 표시 -------------------
if 'result' in st.session_state and 'df_price' in st.session_state and 'df_1y' in st.session_state:
    result = st.session_state['result']
    df_price = st.session_state['df_price']
    df_1y = st.session_state['df_1y']

    st.markdown(
        f"""
        <div style='background-color:#f0f0f0; padding:8px; border-radius:4px; display:inline-block;'>
            <b>기간:</b> {result['start_date'].strftime('%Y-%m-%d')} ~ {result['end_date'].strftime('%Y-%m-%d')} &nbsp; | &nbsp; <b>검색단위:</b> {result['timeframe']}
        </div>
        """,
        unsafe_allow_html=True
    )

    if not df_price.empty and '시가' in df_price.columns and '종가' in df_price.columns and '날짜' in df_price.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_price['날짜'],
            y=df_price['종가'].astype(float),
            mode='lines',
            line=dict(color='lightgray', width=1.2, shape='spline'),
            name='종가(연결선)',
            hoverinfo='skip',
            showlegend=False
        ))

        # 시가-종가 연결선 굵기: Day면 50%, 아니면 100%
        if result['timeframe'] == "Day":
            oc_line_width = 3.75  # 7.5의 50%
        else:
            oc_line_width = 7.5

        # 고가-저가 연결선(색상: 시가-종가 연결선과 동일)
        for i in range(len(df_price)):
            rowp = df_price.iloc[i]
            try:
                open_price = float(str(rowp['시가']).replace(',',''))
                close_price = float(str(rowp['종가']).replace(',',''))
                high = float(str(rowp['고가']).replace(',',''))
                low = float(str(rowp['저가']).replace(',',''))
            except:
                continue
            color = 'red' if open_price < close_price else 'blue'
            fig.add_trace(go.Scatter(
                x=[rowp['날짜'], rowp['날짜']],
                y=[low, high],
                mode='lines',
                line=dict(color=color, width=1.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # 시가-종가 연결선
        for i in range(len(df_price)):
            rowp = df_price.iloc[i]
            try:
                open_price = float(str(rowp['시가']).replace(',',''))
                close_price = float(str(rowp['종가']).replace(',',''))
            except:
                continue
            color = 'red' if open_price < close_price else 'blue'
            fig.add_trace(go.Scatter(
                x=[rowp['날짜'], rowp['날짜']],
                y=[open_price, close_price],
                mode='lines',
                line=dict(color=color, width=oc_line_width),
                showlegend=False,
                hoverinfo='x+y'
            ))

        # 종가 기준 최고/최저가 텍스트 위치 조정 (오프셋 5%)
        closes = df_price['종가'].astype(float)
        max_idx = closes.idxmax()
        min_idx = closes.idxmin()
        max_date = df_price.loc[max_idx, '날짜']
        min_date = df_price.loc[min_idx, '날짜']
        max_val = closes[max_idx]
        min_val = closes[min_idx]
        y_range = max_val - min_val if max_val != min_val else max_val * 0.05
        offset = y_range * 0.05  # 5%로 더 크게

        fig.add_trace(go.Scatter(
            x=[max_date],
            y=[max_val + offset],  # 최고가는 더 위
            mode='text',
            text=[f"<b><span style='color:red'>{int(max_val):,}원</span></b>"],
            textfont=dict(color='red', size=13, family="Nanum Gothic"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[min_date],
            y=[min_val - offset],  # 최저가는 더 아래
            mode='text',
            text=[f"<b><span style='color:blue'>{int(min_val):,}원</span></b>"],
            textfont=dict(color='blue', size=13, family="Nanum Gothic"),
            showlegend=False
        ))

        # 좌측/하단 테두리(밝은 회색) 추가
        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=0, y1=1,
            line=dict(color="#e0e0e0", width=2),
            layer="above"
        )
        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=0,
            line=dict(color="#e0e0e0", width=2),
            layer="above"
        )

        fig.update_layout(
            title=f"{result['selected_name']} ({result['code']}) 시가·종가 추이",
            font=dict(family="Nanum Gothic, NanumGothic, Arial", size=13),
            height=500,
            margin=dict(l=20, r=20, t=60, b=40),
        )
        fig.update_xaxes(type='category', showgrid=False, zeroline=False, title=None)
        fig.update_yaxes(
            tickformat=',',
            tickprefix='',
            ticksuffix='원',
            showgrid=False,
            zeroline=False,
            title=None
        )
        st.plotly_chart(fig, use_container_width=True,config={"displayModeBar": False,"staticPlot": True})

        st.subheader("년간 변동추이")
        df_1y['월'] = df_1y['날짜'].str[:7]
        xlabels = []
        last_month = None
        for v in df_1y['월']:
            if v != last_month:
                xlabels.append(v)
                last_month = v
            else:
                xlabels.append('')
        fig1y = go.Figure()
        fig1y.add_trace(go.Scatter(
            x=df_1y['날짜'],
            y=df_1y['종가'].astype(float),
            mode='lines',
            line=dict(color='lightgray', width=1.2, shape='spline'),
            name='종가(연결선)',
            hoverinfo='skip',
            showlegend=False
        ))
        oc_line_width_1y = 4.5
        for i in range(len(df_1y)):
            rowp = df_1y.iloc[i]
            try:
                open_price = float(str(rowp['시가']).replace(',',''))
                close_price = float(str(rowp['종가']).replace(',',''))
                high = float(str(rowp['고가']).replace(',',''))
                low = float(str(rowp['저가']).replace(',',''))
            except:
                continue
            color = 'red' if open_price < close_price else 'blue'
            fig1y.add_trace(go.Scatter(
                x=[rowp['날짜'], rowp['날짜']],
                y=[low, high],
                mode='lines',
                line=dict(color=color, width=1.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        for i in range(len(df_1y)):
            rowp = df_1y.iloc[i]
            try:
                open_price = float(str(rowp['시가']).replace(',',''))
                close_price = float(str(rowp['종가']).replace(',',''))
            except:
                continue
            color = 'red' if open_price < close_price else 'blue'
            fig1y.add_trace(go.Scatter(
                x=[rowp['날짜'], rowp['날짜']],
                y=[open_price, close_price],
                mode='lines',
                line=dict(color=color, width=oc_line_width_1y),
                showlegend=False,
                hoverinfo='x+y'
            ))
        closes_1y = df_1y['종가'].astype(float)
        max_idx1y = closes_1y.idxmax()
        min_idx1y = closes_1y.idxmin()
        max_date1y = df_1y.loc[max_idx1y, '날짜']
        min_date1y = df_1y.loc[min_idx1y, '날짜']
        max_val1y = closes_1y[max_idx1y]
        min_val1y = closes_1y[min_idx1y]
        y_range1y = max_val1y - min_val1y if max_val1y != min_val1y else max_val1y * 0.05
        offset1y = y_range1y * 0.05  # 5%로 더 크게

        fig1y.add_trace(go.Scatter(
            x=[max_date1y],
            y=[max_val1y + offset1y],
            mode='text',
            text=[f"<b><span style='color:red'>{int(max_val1y):,}원</span></b>"],
            textfont=dict(color='red', size=13, family="Nanum Gothic"),
            showlegend=False
        ))
        fig1y.add_trace(go.Scatter(
            x=[min_date1y],
            y=[min_val1y - offset1y],
            mode='text',
            text=[f"<b><span style='color:blue'>{int(min_val1y):,}원</span></b>"],
            textfont=dict(color='blue', size=13, family="Nanum Gothic"),
            showlegend=False
        ))

        # 좌측/하단 테두리(밝은 회색) 추가 (연간 변동추이 그래프)
        fig1y.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=0, y1=1,
            line=dict(color="#e0e0e0", width=2),
            layer="above"
        )
        fig1y.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=0,
            line=dict(color="#e0e0e0", width=2),
            layer="above"
        )

        fig1y.update_layout(
            title=f"{result['selected_name']} ({result['code']}) 년간 변동추이",
            font=dict(family="Nanum Gothic, NanumGothic, Arial", size=13),
            height=500,
            margin=dict(l=20, r=20, t=60, b=40),
        )
        fig1y.update_xaxes(type='category', showgrid=False, zeroline=False, tickvals=df_1y['날짜'], ticktext=xlabels, title=None)
        fig1y.update_yaxes(
            tickformat=',',
            tickprefix='',
            ticksuffix='원',
            showgrid=False,
            zeroline=False,
            title=None
        )
        st.plotly_chart(fig1y, use_container_width=True,config={"displayModeBar": False,"staticPlot": True})
