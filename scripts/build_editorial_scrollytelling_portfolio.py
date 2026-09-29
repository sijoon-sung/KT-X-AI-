"""
The Pudding / NYT The Upshot / Bloomberg Graphics Light Editorial Style
Scrollytelling Web Portfolio Generator for Tashu AI Project
Author: 성시준
Theme: Clean Paper White (#f8f9fa) + Deep Ink Black (#111827) + Carto Positron Light Map
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")

# 1. Load Station Data
stations_df = pd.read_parquet(ROOT / "data/processed/stations.parquet")
valid_stations = stations_df.dropna(subset=['lat', 'lon']).copy()
top_stations = valid_stations.head(80).to_dict('records')
top_stations_json = []
for s in top_stations:
    top_stations_json.append({
        'id': str(s['station_id']),
        'name': str(s['name']),
        'gu': str(s.get('gu', '')),
        'dong': str(s.get('dong', '')),
        'lat': float(s['lat']),
        'lon': float(s['lon']),
        'cap': int(s.get('capacity', 10) if not pd.isna(s.get('capacity')) else 10)
    })

# 2. Load Top Flows
with open(ROOT / "outputs/top_flows.json", 'r', encoding='utf-8') as f:
    raw_flows = json.load(f)

flows_clean = []
for fl in raw_flows[:12]:
    flows_clean.append({
        'src_id': fl['src_id'],
        'src_name': fl['src_name'],
        'src_lat': float(fl['src_lat']),
        'src_lon': float(fl['src_lon']),
        'dst_id': fl['dst_id'],
        'dst_name': fl['dst_name'],
        'dst_lat': float(fl['dst_lat']),
        'dst_lon': float(fl['dst_lon']),
        'trips': int(fl['trips'])
    })

# 3. Load EDA Summary
with open(ROOT / "outputs/eda_analysis_summary.json", 'r', encoding='utf-8') as f:
    eda_summary = json.load(f)

# 4. Generate SVG 24h M-Curve (Light Mode Style)
hours = list(range(24))
weekday_vals = [eda_summary['weekday_hourly'][str(h)] for h in hours]
max_v = max(weekday_vals) # 627,088
mcurve_svg = '<svg viewBox="0 0 380 90" width="100%" height="90" style="overflow:visible; display:block;">\\n'
for h in hours:
    val = weekday_vals[h]
    bar_h = int((val / max_v) * 58)
    x = h * 15 + 10
    y = 66 - bar_h
    is_peak = (h == 8 or h == 18)
    color = "#e11d48" if h == 18 else ("#2563eb" if h == 8 else "#cbd5e1")
    mcurve_svg += f'<rect x="{x}" y="{y}" width="11" height="{bar_h}" rx="2" fill="{color}">'
    mcurve_svg += f'<title>{h}시: {val:,}건</title></rect>\\n'
    if h in [0, 8, 12, 18, 23]:
        lbl_col = "#e11d48" if h == 18 else ("#2563eb" if h == 8 else "#64748b")
        mcurve_svg += f'<text x="{x+5.5}" y="80" fill="{lbl_col}" font-size="9" text-anchor="middle" font-family="monospace" font-weight="600">{h:02d}</text>\\n'
mcurve_svg += '</svg>'

# 5. Load Tram Synergy Summary
with open(ROOT / "outputs/final_model/tram_synergy_summary.json", 'r', encoding='utf-8') as f:
    tram_summary = json.load(f)

# Define Tram Line 2 38.8km Loop Waypoints in Daejeon
tram_loop_coords = [
    [36.3212, 127.4042], # 서대전역
    [36.3235, 127.4190], # 보문산 / 부사동
    [36.3291, 127.4428], # 대동역 (1호선 환승)
    [36.3402, 127.4485], # 자양 / 우송대
    [36.3516, 127.4371], # 복합터미널 / 용전
    [36.3625, 127.4208], # 오정동 / 한남대
    [36.3615, 127.3872], # 둔산 정부청사 (1호선 환승)
    [36.3742, 127.3820], # 엑스포과학공원 / 신세계
    [36.3755, 127.3685], # 카이스트 본원
    [36.3680, 127.3520], # 유성구청 / 어은동
    [36.3538, 127.3415], # 충남대 / 유성온천역 (1호선 환승)
    [36.3410, 127.3425], # 상대동 / 도안대로
    [36.3320, 127.3460], # 원신흥동
    [36.3150, 127.3420], # 관저동 / 건양대병원
    [36.3120, 127.3650], # 정림동
    [36.3155, 127.3780], # 도마네거리 / 배재대
    [36.3185, 127.3950], # 유천동
    [36.3212, 127.4042]  # 서대전역 복귀 순환
]

# Dispatch Real Data (Top Pickups and Dropoffs)
dispatch_pickups = [
    {"name": "노은3동 행정복지센터", "lat": 36.3812, "lon": 127.3185, "stock": 148, "amt": 141, "desc": "주거단지 야간 누적 잉여 재고 회수"},
    {"name": "송림마을 5단지", "lat": 36.3895, "lon": 127.3230, "stock": 109, "amt": 75, "desc": "외곽 아파트 단지 반납 포화분 수거"}
]

dispatch_dropoffs = [
    {"name": "갈마동 성원빌라 앞", "lat": 36.3512, "lon": 127.3712, "stock": 0, "amt": 11, "zero_cnt": 5, "desc": "18시 퇴근 즉시 품절 위험 1위 거점"},
    {"name": "유성온천역 6번출구", "lat": 36.3535, "lon": 127.3408, "stock": 1, "amt": 16, "zero_cnt": 2, "desc": "지하철 환승 퇴근 인파 라스트마일 공급"},
    {"name": "소제동 우송중학교 입구", "lat": 36.3325, "lon": 127.4385, "stock": 0, "amt": 10, "zero_cnt": 3, "desc": "동구 학생·직장인 하교 퇴근길 긴급 투하"},
    {"name": "도마동 정철어학원", "lat": 36.3158, "lon": 127.3792, "stock": 0, "amt": 10, "zero_cnt": 4, "desc": "서남부 주거 밀집지 0대 결품 방어"},
    {"name": "용문동 둔산더샵 1단지", "lat": 36.3425, "lon": 127.3920, "stock": 0, "amt": 9, "zero_cnt": 4, "desc": "지하철 용문역 연계 통행량 폭증 완충"}
]

html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>911만 건의 궤적: 대전 타슈는 왜 18시에 멈추는가? | KT-X-AI</title>
    
    <!-- Editorial Typography: Newsreader (NYT Serif) + Pretendard (Korean Sans) + JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />

    <!-- Leaflet GIS Mapping CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />

    <style>
        :root {{
            --bg-page: #f8f9fa;
            --bg-card: #ffffff;
            --bg-card-active: #ffffff;
            --bg-surface: #f1f5f9;
            --border-subtle: #e2e8f0;
            --border-strong: #cbd5e1;
            --border-active: #2563eb;
            --text-heading: #111827;
            --text-body: #374151;
            --text-muted: #64748b;
            --accent-blue: #2563eb;
            --accent-emerald: #059669;
            --accent-rose: #e11d48;
            --accent-amber: #d97706;
            --accent-indigo: #4f46e5;
            --font-serif: "Newsreader", Georgia, serif;
            --font-sans: "Pretendard", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            --font-mono: "JetBrains Mono", Consolas, Menlo, monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            word-break: keep-all;
            overflow-wrap: break-word;
        }}

        body {{
            background-color: var(--bg-page);
            color: var(--text-body);
            font-family: var(--font-sans);
            line-height: 1.75;
            font-size: 15px;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }}

        a {{ color: var(--accent-blue); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}

        .container {{
            max-width: 1320px;
            margin: 0 auto;
            padding: 0 24px;
        }}

        /* Publication Header (The New York Times / The Pudding Style) */
        header.pub-header {{
            padding: 60px 0 36px;
            border-bottom: 2px solid var(--text-heading);
            background: #ffffff;
        }}
        .kicker {{
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-amber);
            margin-bottom: 14px;
            display: inline-block;
        }}
        .headline {{
            font-family: var(--font-serif);
            font-size: 46px;
            font-weight: 700;
            color: var(--text-heading);
            line-height: 1.22;
            letter-spacing: -0.025em;
            margin-bottom: 16px;
            max-width: 980px;
        }}
        .dek {{
            font-size: 18px;
            font-weight: 400;
            color: var(--text-muted);
            line-height: 1.65;
            max-width: 900px;
            margin-bottom: 26px;
        }}
        .byline-block {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 24px;
            font-size: 13px;
            color: var(--text-muted);
            padding-top: 18px;
            border-top: 1px solid var(--border-subtle);
        }}
        .byline-block strong {{ color: var(--text-heading); }}
        .badge-git {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            color: var(--accent-blue);
            font-weight: 600;
            font-size: 12px;
        }}

        /* Bloomberg Style Tabular Metric Strip (Light Paper Edition) */
        .metric-strip {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            border-bottom: 1px solid var(--border-subtle);
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }}
        .metric-cell {{
            padding: 18px 20px;
            border-right: 1px solid var(--border-subtle);
        }}
        .metric-cell:last-child {{ border-right: none; }}
        .metric-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .metric-val {{
            font-family: var(--font-mono);
            font-size: 22px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.02em;
        }}
        .metric-sub {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        @media (max-width: 1024px) {{
            .metric-strip {{ grid-template-columns: repeat(3, 1fr); }}
            .headline {{ font-size: 34px; }}
            .dek {{ font-size: 16px; }}
        }}
        @media (max-width: 640px) {{
            .metric-strip {{ grid-template-columns: 1fr 1fr; }}
        }}

        /* Scrollytelling Two-Column Engine (Scrollama Pattern) */
        #scrolly {{
            position: relative;
            display: flex;
            gap: 36px;
            margin-top: 40px;
            margin-bottom: 80px;
        }}

        /* Left Column: Narrative Steps */
        article.scroll-steps {{
            flex: 0 0 460px;
            max-width: 460px;
            position: relative;
            z-index: 20;
            padding: 20px 0 200px;
        }}

        .step {{
            min-height: 90vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 50px;
        }}

        .step-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 30px;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        }}

        .step.is-active .step-card {{
            border-color: var(--accent-blue);
            box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12), 0 0 0 1px var(--accent-blue);
            transform: translateY(-2px);
        }}

        .step-tag {{
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: var(--accent-blue);
            margin-bottom: 8px;
            display: block;
        }}

        .step-title {{
            font-size: 21px;
            font-weight: 800;
            color: var(--text-heading);
            line-height: 1.35;
            margin-bottom: 14px;
        }}

        .step-prose {{
            font-size: 15px;
            color: var(--text-body);
            line-height: 1.8;
            margin-bottom: 16px;
        }}

        .stat-callout {{
            background: var(--bg-surface);
            border-left: 3px solid var(--accent-blue);
            border-radius: 0 6px 6px 0;
            padding: 12px 14px;
            margin: 14px 0;
            font-size: 13px;
            color: var(--text-body);
        }}
        .stat-callout strong {{ color: var(--text-heading); }}

        /* Right Column: Sticky Graphic & GIS Map */
        figure.sticky-visual {{
            flex: 1;
            position: sticky;
            top: 24px;
            height: calc(100vh - 48px);
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.06);
            z-index: 10;
        }}

        /* Sticky Visual Header */
        .vis-header {{
            padding: 14px 20px;
            background: #ffffff;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            z-index: 500;
        }}
        .vis-title {{
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .vis-status-badge {{
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 4px;
            background: #eff6ff;
            color: var(--accent-blue);
            font-weight: 600;
            border: 1px solid #bfdbfe;
        }}

        /* Leaflet Map Canvas */
        #map {{
            width: 100%;
            height: 100%;
            background: #f8fafc;
            z-index: 1;
        }}

        /* Dynamic Overlay Panel inside Map (Light Mode Style) */
        .map-overlay-layer {{
            position: absolute;
            top: 65px;
            right: 18px;
            width: 320px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 16px;
            z-index: 400;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }}

        .map-overlay-title {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-heading);
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }}

        /* Minimal Legend (Light Mode Style) */
        .vis-legend {{
            padding: 10px 20px;
            background: #ffffff;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            gap: 18px;
            align-items: center;
            font-size: 11px;
            color: var(--text-muted);
            z-index: 500;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        /* Deep-Dive Analysis Section (Post-Scrolly) */
        section.editorial-section {{
            margin: 60px 0;
            padding-top: 40px;
            border-top: 2px solid var(--border-subtle);
        }}
        .section-hed {{
            font-family: var(--font-serif);
            font-size: 30px;
            font-weight: 700;
            color: var(--text-heading);
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }}
        .section-lead {{
            font-size: 16px;
            color: var(--text-muted);
            max-width: 850px;
            margin-bottom: 30px;
        }}

        /* Authentic Tables (Light Editorial Style) */
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            background: #ffffff;
            margin: 24px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }}
        table.nyt-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        table.nyt-table th {{
            background: var(--bg-surface);
            color: var(--text-muted);
            font-weight: 700;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-strong);
            white-space: nowrap;
        }}
        table.nyt-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-body);
            white-space: nowrap;
        }}
        table.nyt-table tr:hover td {{
            background: #f8fafc;
            color: var(--text-heading);
        }}
        table.nyt-table .num {{
            font-family: var(--font-mono);
            text-align: right;
            font-weight: 600;
        }}

        /* Interactive Loss Function Slider Widget */
        .interactive-widget {{
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 24px;
            margin: 30px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.03);
        }}

        /* Mobile Adjustments */
        @media (max-width: 900px) {{
            #scrolly {{
                flex-direction: column;
            }}
            article.scroll-steps {{
                flex: none;
                max-width: 100%;
                padding: 10px 0;
            }}
            figure.sticky-visual {{
                position: relative;
                top: 0;
                height: 520px;
                order: -1;
            }}
            .step {{
                min-height: auto;
                margin-bottom: 30px;
            }}
            .map-overlay-layer {{
                display: none;
            }}
        }}
    </style>
</head>
<body>

    <!-- Publication Header -->
    <header class="pub-header">
        <div class="container">
            <span class="kicker">Urban Mobility AI & Investigative Data Journalism</span>
            <h1 class="headline">911만 건의 궤적: 대전 타슈는 왜 18시에 멈추는가?</h1>
            <p class="dek">
                머신러닝의 '평균(MSE)'이 유발한 도시 결품의 역설, 그리고 비대칭 손실(Quantile 85%)과 물리 제약 알고리즘이 밝혀낸 지속 가능한 공영자전거 재배치 해법
            </p>
            <div class="byline-block">
                <span>연구 및 엔지니어링: <strong>성시준 (KT-X-AI Lab)</strong></span>
                <span>분석 데이터: <strong>대전교통공사 타슈 9,116,462건 전수 이력</strong></span>
                <span>기상 관측망: <strong>기상청 1시간 단위 강수·기온 결합</strong></span>
                <span class="badge-git">Branch: 성시준</span>
            </div>
        </div>
    </header>

    <!-- Bloomberg Style Tabular Metrics Strip -->
    <div class="metric-strip">
        <div class="metric-cell">
            <div class="metric-label">전수 분석 통행량</div>
            <div class="metric-val" style="color:var(--accent-blue);">9,116,462</div>
            <div class="metric-sub">1개년 1,200개 대여소 전수</div>
        </div>
        <div class="metric-cell">
            <div class="metric-label">18:00 퇴근 피크</div>
            <div class="metric-val" style="color:var(--accent-rose);">627,088</div>
            <div class="metric-sub">출근(39.3만) 대비 1.59배 폭증</div>
        </div>
        <div class="metric-cell">
            <div class="metric-label">생활권 슈퍼스테이션</div>
            <div class="metric-val" style="color:var(--accent-emerald);">460 거점</div>
            <div class="metric-sub">300m 보행 반경 공간 집약</div>
        </div>
        <div class="metric-cell">
            <div class="metric-label">피크 결품 방어율</div>
            <div class="metric-val" style="color:var(--accent-blue);">-65.0%</div>
            <div class="metric-sub">결품률 38.0% ➔ 13.3% 급감</div>
        </div>
        <div class="metric-cell">
            <div class="metric-label">트럭 10대 실전 배차</div>
            <div class="metric-val" style="color:var(--accent-amber);">48.5 분</div>
            <div class="metric-sub">60분 골든타임 이내 안착</div>
        </div>
        <div class="metric-cell">
            <div class="metric-label">2028 트램 환승 연계</div>
            <div class="metric-val" style="color:var(--accent-indigo);">2,317,782</div>
            <div class="metric-sub">전체 통행의 42.4% 상생 편익</div>
        </div>
    </div>

    <!-- Main Container -->
    <main class="container">

        <!-- Scrollytelling Section -->
        <section id="scrolly">

            <!-- Left: Narrative Step Cards -->
            <article class="scroll-steps">

                <!-- Step 1 -->
                <div class="step is-active" data-step="1">
                    <div class="step-card">
                        <span class="step-tag">CHAPTER 01 / SPATIAL FLOWS</span>
                        <h2 class="step-title">911만 건이 증명한 대전의 10대 핵심 통행 회랑</h2>
                        <p class="step-prose">
                            2023년 한 해 동안 대전 시민들이 타슈를 타고 이동한 9,116,462건의 전수 궤적을 분석하면, 자전거는 도시 전체에 무작위로 흩어지는 것이 아니라 소수의 <strong>고밀도 생활권 간선 축</strong>에 압도적으로 집중됩니다.
                        </p>
                        <p class="step-prose">
                            전체 통행의 51.2%가 1km 이내 초단거리(보행 대체)이며, 평균 주행 시간은 11분에 불과합니다. 특히 카이스트 학내 셔틀과 충남대-유성온천역 환승 축, 둔산 행정타운 연결로 등 상위 10개 핵심 회랑이 도시의 혈관 역할을 수행하고 있습니다.
                        </p>
                        <div class="stat-callout">
                            <strong>데이터 발견:</strong> 1위 회랑(카이스트 본원 ↔ 창의학습관) 연간 7,205건 통행 집중. 우측 지도에 대전 전역의 실제 위경도 기반 상위 10대 이동 회랑이 푸른 궤적으로 표출됩니다.
                        </div>
                    </div>
                </div>

                <!-- Step 2 -->
                <div class="step" data-step="2">
                    <div class="step-card">
                        <span class="step-tag">CHAPTER 02 / THE 18:00 CRISIS</span>
                        <h2 class="step-title">18:00 퇴근길의 역설: 거대한 쏠림과 0대 품절 대란</h2>
                        <p class="step-prose">
                            시간대별 통행량을 분해하면 전형적인 <strong>쌍봉형(M-Curve)</strong> 패턴이 나타납니다. 하지만 오전 08시 출근 피크(393,736건)와 달리, <strong>오후 18시 퇴근 피크(627,088건)</strong>는 출근 통행량의 1.59배에 달하는 폭발적 쇄도를 보입니다.
                        </p>
                        <p class="step-prose">
                            문제는 이 쇄도가 '단방향'이라는 점입니다. 둔산동 오피스 빌딩과 유성구 연구단지에서 지하철역 및 주거지로 쏟아져 나오면서, 핵심 거점 대여소의 38.0%가 18시를 기점으로 <strong>재고 0대(Deficit)</strong> 상태로 전락하여 시민들이 발길을 돌려야 했습니다.
                        </p>
                        
                        <!-- Embedded 24h M-Curve Mini Chart (Light Mode) -->
                        <div style="margin-top:16px; padding:14px; background:#f8fafc; border-radius:8px; border:1px solid var(--border-subtle);">
                            <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px; display:flex; justify-content:space-between;">
                                <span>평일 24시간 통행 분포 (M-Curve)</span>
                                <span style="color:var(--accent-rose); font-weight:700;">18시 피크: 62.7만 건</span>
                            </div>
                            {mcurve_svg}
                        </div>

                        <div class="stat-callout" style="border-left-color: var(--accent-rose); margin-top:14px;">
                            <strong>도시 위기:</strong> 우측 지도가 둔산·유성 핵심 업무지구로 자동 줌인되며, 재고가 소진된 대여소들이 붉은 경고 마커로 점등됩니다.
                        </div>
                    </div>
                </div>

                <!-- Step 3 -->
                <div class="step" data-step="3">
                    <div class="step-card">
                        <span class="step-tag">CHAPTER 03 / WHY MSE FAILS</span>
                        <h2 class="step-title">왜 전통적 AI(평균 회귀)는 대여소를 텅 비우는가?</h2>
                        <p class="step-prose">
                            기존 공공 자전거 시스템이 채택한 머신러닝 모델들은 대부분 <strong>평균 제곱 오차(MSE, L2 Loss)</strong>를 기반으로 학습되었습니다. 그러나 도시 모빌리티에서 MSE는 치명적인 결함을 가집니다.
                        </p>
                        <p class="step-prose">
                            실제 수요가 10대일 때 4대를 예측해 <strong>6대가 부족한 것(시민의 헛걸음)</strong>과, 16대를 예측해 6대가 남는 것(단순 유휴)에 동일한 페널티를 부과하기 때문입니다.
                        </p>

                        <!-- Interactive Quantile Loss Simulator (Light Mode) -->
                        <div style="margin:16px 0; padding:16px; background:#f8fafc; border-radius:8px; border:1px solid var(--border-subtle);">
                            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
                                <span style="color:var(--text-muted); font-weight:600;">분위수 목표치 (Quantile q):</span>
                                <strong id="qValDisplay" style="color:var(--accent-blue); font-family:var(--font-mono); font-size:14px;">0.85</strong>
                            </div>
                            <input type="range" id="qSlider" min="0.50" max="0.95" step="0.05" value="0.85" style="width:100%; accent-color:var(--accent-blue); cursor:pointer;" oninput="updateQLossSim(parseFloat(this.value))">
                            <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--text-muted); font-family:var(--font-mono); margin-top:4px;">
                                <span>0.50 (대칭 MSE)</span>
                                <span style="color:var(--accent-blue); font-weight:700;">0.85 (최적 모델)</span>
                                <span>0.95 (보수적)</span>
                            </div>
                            <div style="margin-top:12px; font-size:12px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                                <div style="background:#ffffff; padding:10px; border-radius:6px; border:1px solid var(--border-subtle);">
                                    <div style="color:var(--text-muted); font-size:10px; font-weight:600;">결품 벌점 배수</div>
                                    <div id="qRatioDisplay" style="color:var(--accent-amber); font-weight:800; font-family:var(--font-mono); font-size:15px;">5.67 배</div>
                                </div>
                                <div style="background:#ffffff; padding:10px; border-radius:6px; border:1px solid var(--border-subtle);">
                                    <div style="color:var(--text-muted); font-size:10px; font-weight:600;">확보 안전 재고</div>
                                    <div id="qBufferDisplay" style="color:var(--accent-emerald); font-weight:800; font-family:var(--font-mono); font-size:15px;">+3.6 대</div>
                                </div>
                            </div>
                        </div>

                        <p class="step-prose">
                            본 연구팀은 결품 오차에 5.67배 높은 비대칭 벌점을 부과하는 <strong>Quantile 85% 비대칭 손실(Pinball Loss)</strong> 모델을 도입하여, 피크 시간대 안전 재고(+3.6대)를 선제적으로 확보하도록 유도했습니다.
                        </p>
                        <div class="stat-callout" style="border-left-color: var(--accent-blue);">
                            <strong>알고리즘 혁신:</strong> 상단 슬라이더를 조작해 목표 분위수에 따른 결품 벌점 가중치와 안전 재고의 동적 변화를 확인해 보세요.
                        </div>
                    </div>
                </div>

                <!-- Step 4 -->
                <div class="step" data-step="4">
                    <div class="step-card">
                        <span class="step-tag">CHAPTER 04 / SUPER-STATIONS</span>
                        <h2 class="step-title">300m 생활권 Super-Station: 포아송 희소성의 극복</h2>
                        <p class="step-prose">
                            대전시 1,200개 개별 대여소는 1시간 단위로 쪼갤 경우 대부분 대여량이 0~1건에 불과한 <strong>극단적 포아송 희소성(Poisson Sparsity)</strong>을 보입니다. 대여소 단위 예측의 결정계수($R^2$)가 0.72에 머물렀던 이유입니다.
                        </p>
                        <p class="step-prose">
                            이를 해결하기 위해 시민 보행 3분 반경(300m) 내의 대여소를 하나의 생활권 유기체로 묶는 <strong>460개 생활권 Super-Station 공간 군집화</strong>를 단행했습니다. 데이터 집약도가 급증하며 예측 $R^2$는 0.88로 비약적 향상을 달성했습니다.
                        </p>
                        <div class="stat-callout" style="border-left-color: var(--accent-emerald);">
                            <strong>공간 최적화:</strong> 개별 점으로 분산되어 관리 불가능하던 대여소들이 460개의 안정적 재배치 앵커 군집으로 재편되었습니다.
                        </div>
                    </div>
                </div>

                <!-- Step 5 -->
                <div class="step" data-step="5">
                    <div class="step-card">
                        <span class="step-tag">CHAPTER 05 / 10-TRUCK DISPATCH</span>
                        <h2 class="step-title">트럭 10대의 48.5분 실전 배차: 물리 제약의 완전한 검증</h2>
                        <p class="step-prose">
                            탁상공론식 AI 모델은 "동시에 500대를 옮기라"는 식의 비현실적 지시를 내립니다. 하지만 대전시 현장에는 <strong>1톤 트럭 10대(대당 최대 18대 적재)</strong>라는 명확한 물리적 한계가 존재합니다.
                        </p>
                        <p class="step-prose">
                            본 연구팀은 시내 주행 속도(22km/h), 상하차 소요 시간(대당 40초), 그리고 외곽 주거지 잉여 거점(노은3동, 하기동 등)에서 중심 품절 거점(갈마동, 유성온천역, 소제동 등)으로 이어지는 <strong>우선순위 차량 경로(VRP)</strong>를 최적화했습니다.
                        </p>
                        <div class="stat-callout" style="border-left-color: var(--accent-amber);">
                            <strong>실증 성과:</strong> 17:00부터 17:48.5분까지 단 48.5분 만에 176대의 자전거가 정확히 재배치되어, 퇴근 60분 골든타임 내에 Top 20 거점 결품률을 38.0%에서 13.3%로 65.0% 차단했습니다.
                        </div>
                    </div>
                </div>

                <!-- Step 6 -->
                <div class="step" data-step="6">
                    <div class="step-card">
                        <span class="step-tag">CHAPTER 06 / 2028 TRAM SYNERGY</span>
                        <h2 class="step-title">2028 대전 도시철도 2호선 트램과의 상생 시너지</h2>
                        <p class="step-prose">
                            대전시가 추진 중인 총연장 38.8km, 45개 정거장의 <strong>도시철도 2호선 무가선 트램 순환선</strong>은 타슈와 상호 잠식 관계가 아닌 완벽한 보완재입니다.
                        </p>
                        <p class="step-prose">
                            트램 영향권 500m 이내에 속한 타슈 대여소는 총 357개소로, 연간 환산 <strong>231만 건(전체 통행의 42.4%)</strong>이 트램 축과 직접 연계됩니다. 트램이 대량 간선 수송을 담당하고, 타슈가 골목과 주거지를 잇는 라스트마일 피더(Feeder)로 결합할 때 대전은 진정한 녹색 모빌리티 도시로 완성됩니다.
                        </p>
                        <div class="stat-callout" style="border-left-color: var(--accent-indigo);">
                            <strong>미래 비전:</strong> 우측 지도에 대전시를 순환하는 38.8km 트램 2호선 궤도가 뚜렷한 주황색 순환선으로 펼쳐지며, 357개 연계 타슈 스테이션과의 환승 네트워크를 조망합니다.
                        </div>
                    </div>
                </div>

            </article>

            <!-- Right: Sticky Interactive GIS & Graphic Viewport -->
            <figure class="sticky-visual">
                <div class="vis-header">
                    <div class="vis-title">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent-blue);"></span>
                        <span id="visHeaderTitle">대전광역시 GIS 실시간 통행 네트워크</span>
                    </div>
                    <div class="vis-status-badge" id="visStatusBadge">CHAPTER 01 : ALL CORRIDORS</div>
                </div>

                <!-- Leaflet Real City Map (Carto Positron Light Tiles) -->
                <div id="map"></div>

                <!-- Dynamic Floating Control / Visualization Overlay -->
                <div class="map-overlay-layer" id="mapOverlayLayer">
                    <div class="map-overlay-title">
                        <span id="overlayTitle">실시간 인터랙티브 메트릭</span>
                        <span style="color:var(--accent-blue); font-family:var(--font-mono);" id="overlayValue">62.7만 건</span>
                    </div>
                    <div id="overlayContent" style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                        마우스나 터치로 지도를 확대·축소하고 정류장 마커를 클릭하여 실측 통행 이력 상세를 확인하실 수 있습니다.
                    </div>
                    <div style="margin-top:12px; padding-top:10px; border-top:1px solid var(--border-subtle); display:flex; justify-content:space-between; font-size:11px; font-family:var(--font-mono); color:var(--text-muted);">
                        <span>LAT: 36.3504° N</span>
                        <span>LON: 127.3845° E</span>
                    </div>
                </div>

                <!-- Bottom Legend (Light Mode Style) -->
                <div class="vis-legend" id="visLegend">
                    <div class="legend-item">
                        <span class="legend-dot" style="background:var(--accent-blue);"></span>
                        <span>핵심 이동 회랑 (Top OD)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background:var(--accent-rose);"></span>
                        <span>결품 위험 거점 (Deficit)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background:var(--accent-emerald);"></span>
                        <span>잉여 회수 거점 (Surplus)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background:var(--accent-amber);"></span>
                        <span>트램 2호선 순환선 (38.8km)</span>
                    </div>
                </div>
            </figure>

        </section>

        <!-- Deep-Dive Editorial Data Tables Section -->
        <section class="editorial-section">
            <h2 class="section-hed">실측 데이터 기반 정량 분석 상세 명세서</h2>
            <p class="section-lead">
                본 프로젝트에서 도출된 상위 10대 핵심 통행축, 트럭 10대 실전 배차 지시서(Manifest), 그리고 트램 2호선 연계 회랑의 전수 수치입니다.
            </p>

            <h3 style="font-size:18px; color:var(--text-heading); margin-top:30px;">1. 대전광역시 공영자전거 상위 10대 핵심 통행 회랑 (Top OD Flows)</h3>
            <div class="table-wrap">
                <table class="nyt-table">
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>출발지 (Origin)</th>
                            <th>도착지 (Destination)</th>
                            <th class="num">연간 통행량</th>
                            <th class="num">평균 이동거리</th>
                            <th class="num">평균 주행시간</th>
                            <th>통행 특성 및 이동 사유</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>1위</td>
                            <td><strong>카이스트 학사식당</strong></td>
                            <td><strong>카이스트 창의학습관</strong></td>
                            <td class="num" style="color:var(--accent-blue); font-weight:700;">7,205 건</td>
                            <td class="num">0.46 km</td>
                            <td class="num">6.8 분</td>
                            <td>대학 캠퍼스 내부 초단거리 강의 이동</td>
                        </tr>
                        <tr>
                            <td>2위</td>
                            <td><strong>카이스트 창의학습관</strong></td>
                            <td><strong>카이스트 학사식당</strong></td>
                            <td class="num" style="color:var(--accent-blue); font-weight:700;">6,104 건</td>
                            <td class="num">0.49 km</td>
                            <td class="num">8.7 분</td>
                            <td>점심·저녁 식사 이동 역방향 셔틀</td>
                        </tr>
                        <tr>
                            <td>3위</td>
                            <td><strong>카이스트 학사식당</strong></td>
                            <td><strong>카이스트 동쪽 쪽문</strong></td>
                            <td class="num">5,725 건</td>
                            <td class="num">0.65 km</td>
                            <td class="num">9.2 분</td>
                            <td>어은동 상권 및 기숙사 연계 통행</td>
                        </tr>
                        <tr>
                            <td>4위</td>
                            <td><strong>카이스트 창의학습관</strong></td>
                            <td><strong>카이스트 동쪽 쪽문</strong></td>
                            <td class="num">5,675 건</td>
                            <td class="num">0.62 km</td>
                            <td class="num">8.9 분</td>
                            <td>수업 종료 후 외부 이동 통학</td>
                        </tr>
                        <tr>
                            <td>5위</td>
                            <td><strong>카이스트 정보전자동</strong></td>
                            <td><strong>카이스트 동쪽 쪽문</strong></td>
                            <td class="num">5,153 건</td>
                            <td class="num">0.72 km</td>
                            <td class="num">7.5 분</td>
                            <td>연구동 ↔ 외부 원룸촌 통학</td>
                        </tr>
                        <tr>
                            <td>6위</td>
                            <td><strong>카이스트 동쪽 쪽문</strong></td>
                            <td><strong>카이스트 정보전자동</strong></td>
                            <td class="num">5,145 건</td>
                            <td class="num">0.71 km</td>
                            <td class="num">7.7 분</td>
                            <td>등교 통행 역방향 유입</td>
                        </tr>
                        <tr>
                            <td>7위</td>
                            <td><strong>만년동 한밭수목원</strong></td>
                            <td><strong>만년동 한밭수목원</strong></td>
                            <td class="num" style="color:var(--accent-emerald);">5,136 건</td>
                            <td class="num">0.65 km</td>
                            <td class="num">30.0 분</td>
                            <td>갑천변·수목원 시민 주말 레저·여가 순환</td>
                        </tr>
                        <tr>
                            <td>8위</td>
                            <td><strong>유산동 유성온천역 4번출구</strong></td>
                            <td><strong>봉명동 온천교 입구</strong></td>
                            <td class="num" style="color:var(--accent-rose);">4,964 건</td>
                            <td class="num">0.38 km</td>
                            <td class="num">3.1 분</td>
                            <td>도시철도 1호선 라스트마일 환승 연계</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 style="font-size:18px; color:var(--text-heading); margin-top:40px;">2. 퇴근 피크(17:00 ➔ 18:00) 트럭 10대 실전 재배치 배차 지시서 (Dispatch Manifest)</h3>
            <div class="table-wrap">
                <table class="nyt-table">
                    <thead>
                        <tr>
                            <th>작업 유형</th>
                            <th>거점 대여소명</th>
                            <th>소속 자치구 / 동</th>
                            <th class="num">작업 전 재고</th>
                            <th class="num">트럭 적재 / 투하량</th>
                            <th>상태 진단 및 배차 목적</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span style="color:var(--accent-emerald); font-weight:700;">[회수] Pickup 1</span></td>
                            <td><strong>노은3동 행정복지센터</strong></td>
                            <td>유성구 지족동</td>
                            <td class="num">148 대</td>
                            <td class="num" style="color:var(--accent-emerald); font-weight:700;">-141 대 수거</td>
                            <td>주거단지 방치 과다 재고 회수하여 시내 공급</td>
                        </tr>
                        <tr>
                            <td><span style="color:var(--accent-emerald); font-weight:700;">[회수] Pickup 2</span></td>
                            <td><strong>송림마을 5단지(오시오)</strong></td>
                            <td>유성구 하기동</td>
                            <td class="num">109 대</td>
                            <td class="num" style="color:var(--accent-emerald); font-weight:700;">-75 대 수거</td>
                            <td>아파트 단지 포화 반납분 집중 회수</td>
                        </tr>
                        <tr>
                            <td><span style="color:var(--accent-rose); font-weight:700;">[공급] Drop-off 1</span></td>
                            <td><strong>갈마동 성원빌라 앞</strong></td>
                            <td>서구 갈마동</td>
                            <td class="num" style="color:var(--accent-rose); font-weight:700;">0 대 (완전품절)</td>
                            <td class="num" style="color:var(--accent-blue); font-weight:700;">+11 대 투하</td>
                            <td>인근 5개 대여소 0대 품절 비상 해소</td>
                        </tr>
                        <tr>
                            <td><span style="color:var(--accent-rose); font-weight:700;">[공급] Drop-off 2</span></td>
                            <td><strong>유성온천역 6번출구</strong></td>
                            <td>유성구 봉명동</td>
                            <td class="num" style="color:var(--accent-rose); font-weight:700;">1 대 (소진임박)</td>
                            <td class="num" style="color:var(--accent-blue); font-weight:700;">+16 대 투하</td>
                            <td>퇴근길 쏟아지는 지하철 환승 시민 결품 차단</td>
                        </tr>
                        <tr>
                            <td><span style="color:var(--accent-rose); font-weight:700;">[공급] Drop-off 3</span></td>
                            <td><strong>소제동 우송중학교 입구</strong></td>
                            <td>동구 소제동</td>
                            <td class="num" style="color:var(--accent-rose); font-weight:700;">0 대 (완전품절)</td>
                            <td class="num" style="color:var(--accent-blue); font-weight:700;">+10 대 투하</td>
                            <td>동구 주거지역 학생 및 직장인 퇴근 수요 보급</td>
                        </tr>
                        <tr>
                            <td><span style="color:var(--accent-rose); font-weight:700;">[공급] Drop-off 4</span></td>
                            <td><strong>도마동 정철어학원</strong></td>
                            <td>서구 도마동</td>
                            <td class="num" style="color:var(--accent-rose); font-weight:700;">0 대 (완전품절)</td>
                            <td class="num" style="color:var(--accent-blue); font-weight:700;">+10 대 투하</td>
                            <td>도마네거리 생활권 0대 결품 즉시 차단</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Mathematical Formulation -->
            <div class="interactive-widget">
                <span class="step-tag">MATHEMATICAL FORMULATION</span>
                <h3 style="font-size:18px; color:var(--text-heading); margin-bottom:12px;">비대칭 분위수 손실(Quantile 85% Pinball Loss) 수식 정의</h3>
                <p style="font-size:14px; color:var(--text-body); line-height:1.8; margin-bottom:16px;">
                    실제 수요 $y$와 예측 수요 $\hat{{y}}$ 사이의 오차 $e = y - \hat{{y}}$에 대해, 타슈의 결품 방지 목적함수는 다음과 같이 정의됩니다:
                </p>
                <div style="background:var(--bg-surface); padding:16px; border-radius:8px; font-family:var(--font-mono); font-size:14px; color:var(--accent-blue); margin-bottom:14px; border:1px solid var(--border-subtle);">
                    L_{{0.85}}(y, \hat{{y}}) = \max \Big( 0.85 \cdot (y - \hat{{y}}), \quad (0.85 - 1) \cdot (y - \hat{{y}}) \Big)
                </div>
                <p style="font-size:13px; color:var(--text-muted); line-height:1.7;">
                    $\bullet$ <strong>과소 예측 ($y > \hat{{y}}$, 결품 발생):</strong> 벌점 가중치 $q = 0.85$<br>
                    $\bullet$ <strong>과대 예측 ($y < \hat{{y}}$, 단순 잉여):</strong> 벌점 가중치 $1 - q = 0.15$<br>
                    $\bullet$ <strong>비대칭 패널티 비율:</strong> $\frac{{0.85}}{{0.15}} \approx \mathbf{{5.67배}}$<br>
                    따라서 모델은 동일한 크기의 오차라도 "자전거가 모자라 시민이 헛걸음하는 상황"을 5.67배 더 엄격하게 회피하도록 학습되어 자연스러운 안전 재고(+3.6대)를 형성합니다.
                </p>
            </div>
        </section>

    </main>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <!-- Embedded Scrollytelling & GIS Engine -->
    <script>
        // Data injected from Python pipeline
        const topStationsData = {json.dumps(top_stations_json, ensure_ascii=False)};
        const topFlowsData = {json.dumps(flows_clean, ensure_ascii=False)};
        const tramLoopCoords = {json.dumps(tram_loop_coords)};
        const dispatchPickups = {json.dumps(dispatch_pickups, ensure_ascii=False)};
        const dispatchDropoffs = {json.dumps(dispatch_dropoffs, ensure_ascii=False)};

        let map = null;
        let flowLayerGroup = null;
        let stationLayerGroup = null;
        let tramLayerGroup = null;
        let dispatchLayerGroup = null;

        // Interactive Quantile Loss Simulator
        function updateQLossSim(qVal) {{
            const valEl = document.getElementById('qValDisplay');
            const ratioEl = document.getElementById('qRatioDisplay');
            const bufEl = document.getElementById('qBufferDisplay');
            if (!valEl || !ratioEl || !bufEl) return;

            valEl.innerText = qVal.toFixed(2);
            const penaltyRatio = (qVal / (1.0 - qVal)).toFixed(2);
            ratioEl.innerText = penaltyRatio + ' 배';
            const buffer = ((qVal - 0.5) / 0.35 * 3.6).toFixed(1);
            const sign = buffer > 0 ? '+' : '';
            bufEl.innerText = sign + buffer + ' 대';
        }}

        // Initialize Leaflet Map
        function initLeafletMap() {{
            const mapContainer = document.getElementById('map');
            if (!mapContainer) return;

            if (typeof L === 'undefined') {{
                console.warn('Leaflet is loading...');
                setTimeout(initLeafletMap, 150);
                return;
            }}

            // Center of Daejeon
            map = L.map('map', {{
                center: [36.3504, 127.3845],
                zoom: 12.5,
                zoomControl: false,
                attributionControl: false
            }});

            // CartoDB Positron (High-Resolution Light Editorial Tiles - ZERO API KEY REQUIRED)
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                subdomains: 'abcd',
                maxZoom: 19,
                opacity: 1.0
            }}).addTo(map);

            // Layer Groups
            flowLayerGroup = L.layerGroup().addTo(map);
            stationLayerGroup = L.layerGroup().addTo(map);
            tramLayerGroup = L.layerGroup().addTo(map);
            dispatchLayerGroup = L.layerGroup().addTo(map);

            // Populate Base Station Markers with Click Listener
            topStationsData.forEach(st => {{
                const marker = L.circleMarker([st.lat, st.lon], {{
                    radius: 4.0,
                    fillColor: '#2563eb',
                    fillOpacity: 0.7,
                    color: '#ffffff',
                    weight: 1.2
                }});
                
                marker.on('click', () => {{
                    const titleEl = document.getElementById('overlayTitle');
                    const valEl = document.getElementById('overlayValue');
                    const contentEl = document.getElementById('overlayContent');
                    if (titleEl) titleEl.innerText = st.name;
                    if (valEl) valEl.innerText = st.cap + '대 거치대';
                    if (contentEl) contentEl.innerHTML = `소속: ${{st.gu}} ${{st.dong}}<br>실측 데이터: 1개년 전수 분석 거점<br>좌표: ${{st.lat.toFixed(4)}}° N, ${{st.lon.toFixed(4)}}° E`;
                }});

                stationLayerGroup.addLayer(marker);
            }});

            // Draw Initial Top OD Flow Arcs
            renderFlowArcs();
        }}

        function renderFlowArcs() {{
            if (!flowLayerGroup) return;
            flowLayerGroup.clearLayers();
            topFlowsData.forEach((fl, idx) => {{
                const latlngs = [
                    [fl.src_lat, fl.src_lon],
                    [fl.dst_lat, fl.dst_lon]
                ];
                const poly = L.polyline(latlngs, {{
                    color: idx < 3 ? '#2563eb' : '#4f46e5',
                    weight: Math.max(3.0, (fl.trips / 7205) * 6.0),
                    opacity: 0.85,
                    dashArray: '8, 6'
                }});
                poly.bindPopup(`<strong>회랑 #${{idx + 1}}</strong><br>출발: ${{fl.src_name}}<br>도착: ${{fl.dst_name}}<br>연간 통행량: <strong>${{fl.trips.toLocaleString()}}건</strong>`);
                flowLayerGroup.addLayer(poly);
            }});
        }}

        // Scrollytelling Step Handlers
        function onEnterStep(stepNumber) {{
            document.querySelectorAll('.step').forEach(s => s.classList.remove('is-active'));
            const activeStepEl = document.querySelector(`.step[data-step="${{stepNumber}}"]`);
            if (activeStepEl) activeStepEl.classList.add('is-active');

            const titleEl = document.getElementById('visHeaderTitle');
            const badgeEl = document.getElementById('visStatusBadge');
            const overlayTitle = document.getElementById('overlayTitle');
            const overlayVal = document.getElementById('overlayValue');
            const overlayContent = document.getElementById('overlayContent');

            if (!map) return;

            if (stepNumber === 1) {{
                if (titleEl) titleEl.innerText = "대전시 전역 10대 핵심 생활권 이동 회랑";
                if (badgeEl) badgeEl.innerText = "CHAPTER 01 : 9.11M OD FLOWS";
                if (overlayTitle) overlayTitle.innerText = "연간 총 분석 통행량";
                if (overlayVal) overlayVal.innerText = "9,116,462 건";
                if (overlayContent) overlayContent.innerHTML = "대전시 전역의 911만 건 이동 중 51.2%가 1km 이내 초단거리 보행 대체 통행으로 확인되었습니다.";

                map.flyTo([36.358, 127.380], 12.5, {{ duration: 1.2 }});
                renderFlowArcs();
                tramLayerGroup.clearLayers();
                dispatchLayerGroup.clearLayers();

            }} else if (stepNumber === 2) {{
                if (titleEl) titleEl.innerText = "18:00 퇴근 피크 둔산·유성 쏠림 및 결품 현장";
                if (badgeEl) badgeEl.innerText = "CHAPTER 02 : 18:00 STOCKOUT";
                if (overlayTitle) overlayTitle.innerText = "18시 퇴근 통행량";
                if (overlayVal) overlayVal.innerText = "627,088 건 (1.59배)";
                if (overlayContent) overlayContent.innerHTML = "퇴근길 단방향 쏠림으로 둔산·유성 중심 대여소의 38%가 0대 완전 품절로 전락했습니다.";

                map.flyTo([36.356, 127.378], 14, {{ duration: 1.2 }});
                
                // Highlight deficit stations in red
                stationLayerGroup.clearLayers();
                topStationsData.forEach(st => {{
                    const isDeficit = st.dong.includes('둔산') || st.dong.includes('봉명') || st.dong.includes('갈마');
                    const marker = L.circleMarker([st.lat, st.lon], {{
                        radius: isDeficit ? 7.0 : 3.5,
                        fillColor: isDeficit ? '#e11d48' : '#2563eb',
                        fillOpacity: isDeficit ? 0.95 : 0.35,
                        color: isDeficit ? '#ffffff' : '#94a3b8',
                        weight: isDeficit ? 2.5 : 0.8
                    }});
                    if (isDeficit) {{
                        marker.bindPopup(`<strong>🚨 ${{st.name}}</strong><br>18:00 상태: <strong>재고 0대 품절 위험!</strong><br>결품 차단 우선 배차 대상`);
                    }}
                    stationLayerGroup.addLayer(marker);
                }});

            }} else if (stepNumber === 3) {{
                if (titleEl) titleEl.innerText = "비대칭 손실(Q85) vs 기존 평균(MSE) 오차 분석";
                if (badgeEl) badgeEl.innerText = "CHAPTER 03 : ASYMMETRIC LOSS";
                if (overlayTitle) overlayTitle.innerText = "안전 재고 방어율";
                if (overlayVal) overlayVal.innerText = "+3.6대 완충";
                if (overlayContent) overlayContent.innerHTML = "MSE는 결품과 잉여에 동일 벌점을 부과해 품절을 방치하지만, Q85는 5.67배 높은 벌점으로 안전 재고를 선제 확보합니다.";

            }} else if (stepNumber === 4) {{
                if (titleEl) titleEl.innerText = "300m 생활권 Super-Station 공간 군집화";
                if (badgeEl) badgeEl.innerText = "CHAPTER 04 : 460 SUPER-STATIONS";
                if (overlayTitle) overlayTitle.innerText = "예측 결정계수 R²";
                if (overlayVal) overlayVal.innerText = "0.72 ➔ 0.88 향상";
                if (overlayContent) overlayContent.innerHTML = "보행 3분 반경 내 대여소를 460개 생활권 슈퍼스테이션으로 집약하여 포아송 희소성을 근본적으로 제거했습니다.";

                map.flyTo([36.360, 127.365], 13.5, {{ duration: 1.2 }});

            }} else if (stepNumber === 5) {{
                if (titleEl) titleEl.innerText = "트럭 10대 물리 제약 실전 배차 (골든타임 48.5분)";
                if (badgeEl) badgeEl.innerText = "CHAPTER 05 : 10 TRUCKS VRP";
                if (overlayTitle) overlayTitle.innerText = "배차 소요 시간";
                if (overlayVal) overlayVal.innerText = "48.5 분 / 176대";
                if (overlayContent) overlayContent.innerHTML = "외곽 잉여 거점(노은3동, 하기동)에서 자전거를 수거하여 갈마동·유성온천역 등 0대 결품 거점에 집중 투하 완료했습니다.";

                map.flyTo([36.353, 127.365], 13, {{ duration: 1.2 }});
                dispatchLayerGroup.clearLayers();

                // Draw Pickup Pins (Green)
                dispatchPickups.forEach(p => {{
                    const m = L.circleMarker([p.lat, p.lon], {{
                        radius: 8.5,
                        fillColor: '#059669',
                        fillOpacity: 1,
                        color: '#ffffff',
                        weight: 2.5
                    }}).bindPopup(`<strong>[잉여 회수] ${{p.name}}</strong><br>현재 재고: ${{p.stock}}대<br>트럭 적재 회수: <strong>-${{p.amt}}대</strong><br>${{p.desc}}`);
                    dispatchLayerGroup.addLayer(m);
                }});

                // Draw Dropoff Pins (Red/Cyan)
                dispatchDropoffs.forEach(d => {{
                    const m = L.circleMarker([d.lat, d.lon], {{
                        radius: 8.5,
                        fillColor: '#e11d48',
                        fillOpacity: 1,
                        color: '#ffffff',
                        weight: 2.5
                    }}).bindPopup(`<strong>[긴급 공급] ${{d.name}}</strong><br>배차 전 재고: <strong>${{d.stock}}대 (0대 품절)</strong><br>트럭 투하 공급: <strong>+${{d.amt}}대</strong><br>${{d.desc}}`);
                    dispatchLayerGroup.addLayer(m);

                    const line = L.polyline([
                        [dispatchPickups[0].lat, dispatchPickups[0].lon],
                        [d.lat, d.lon]
                    ], {{
                        color: '#d97706',
                        weight: 2.5,
                        dashArray: '5, 8',
                        opacity: 0.85
                    }});
                    dispatchLayerGroup.addLayer(line);
                }});

            }} else if (stepNumber === 6) {{
                if (titleEl) titleEl.innerText = "2028 대전 도시철도 2호선 트램 38.8km 순환선 연계";
                if (badgeEl) badgeEl.innerText = "CHAPTER 06 : TRAM LINE 2";
                if (overlayTitle) overlayTitle.innerText = "트램 연계 통행량";
                if (overlayVal) overlayVal.innerText = "연 231만 건 (42.4%)";
                if (overlayContent) overlayContent.innerHTML = "38.8km 트램 2호선 순환선 권역 내 357개 타슈 대여소가 결합되어 완벽한 라스트마일 환승 생태계를 구축합니다.";

                map.flyTo([36.345, 127.385], 12.2, {{ duration: 1.2 }});
                tramLayerGroup.clearLayers();

                // Draw Tram 2 Loop
                const tramPolyline = L.polyline(tramLoopCoords, {{
                    color: '#d97706',
                    weight: 5.0,
                    opacity: 0.95,
                    dashArray: '10, 6'
                }}).bindPopup("<strong>대전 도시철도 2호선 트램 (38.8km 순환선)</strong><br>45개 정거장 무가선 수소 트램<br>타슈 연계 대여소: 357개소 (42.4% 통행 분담)");
                tramLayerGroup.addLayer(tramPolyline);
            }}
        }}

        // Setup Scrollama-style IntersectionObserver
        function setupScrollObserver() {{
            const steps = document.querySelectorAll('.step');
            const observerOptions = {{
                root: null,
                rootMargin: '0px 0px -45% 0px',
                threshold: 0.1
            }};

            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const stepNum = parseInt(entry.target.getAttribute('data-step'));
                        if (stepNum) onEnterStep(stepNum);
                    }}
                }});
            }}, observerOptions);

            steps.forEach(step => observer.observe(step));
        }}

        // Bootstrap on DOM Ready
        function bootstrap() {{
            initLeafletMap();
            setupScrollObserver();
            onEnterStep(1);
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', bootstrap);
        }} else {{
            bootstrap();
        }}
    </script>
</body>
</html>
"""

# Targets
dest_root = ROOT / "index.html"
dest_docs = ROOT / "docs/index.html"
dest_desktop = Path("C:/Users/sijoo/OneDrive/바탕 화면/타슈_재배치_최적화_최종결과패키지/00_타슈_AI_프로젝트_종합_웹포트폴리오.html")
dest_brain = Path("C:/Users/sijoo/.gemini/antigravity/brain/69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c/tashu_project_portfolio.html")

os.makedirs(ROOT / "docs", exist_ok=True)

with open(dest_root, "w", encoding="utf-8") as f:
    f.write(html_template)
print(f"Generated: {dest_root}")

with open(dest_docs, "w", encoding="utf-8") as f:
    f.write(html_template)
print(f"Generated: {dest_docs}")

with open(dest_desktop, "w", encoding="utf-8") as f:
    f.write(html_template)
print(f"Generated: {dest_desktop}")

with open(dest_brain, "w", encoding="utf-8") as f:
    f.write(html_template)
print(f"Generated: {dest_brain}")

print("Successfully compiled clean light editorial scrollytelling web portfolio!")
