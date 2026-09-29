import base64
import json
import os
from pathlib import Path

ROOT = Path("C:/Users/sijoo/Documents/tashu")

# 1. Load EDA Summary
with open(ROOT / "outputs/eda_analysis_summary.json", "r", encoding="utf-8") as f:
    eda = json.load(f)

# 2. Encode f5_odmap.png to base64
od_map_b64 = ""
od_map_path = ROOT / "outputs/report_figs/f5_odmap.png"
if od_map_path.exists():
    with open(od_map_path, "rb") as f:
        od_map_b64 = base64.b64encode(f.read()).decode("utf-8")

# 3. Prepare Hourly Data for SVG Chart
weekday_hours = [eda["weekday_hourly"][str(h)] for h in range(24)]
weekend_hours = [eda["weekend_hourly"][str(h)] for h in range(24)]

max_val = 650000  # For chart scale

# SVG Dimensions
chart_w = 940
chart_h = 260
margin_left = 60
margin_bottom = 40
plot_w = chart_w - margin_left - 20
plot_h = chart_h - margin_bottom - 20

# Generate SVG Grid Lines and Y-Axis Ticks
y_ticks_svg = ""
for tick in [0, 100000, 200000, 300000, 400000, 500000, 600000]:
    y = 20 + plot_h - (tick / max_val) * plot_h
    label = f"{tick // 10000}만" if tick > 0 else "0"
    y_ticks_svg += f'<line x1="{margin_left}" y1="{y}" x2="{chart_w - 20}" y2="{y}" stroke="#1e293b" stroke-width="1" stroke-dasharray="{"0" if tick==0 else "3 3"}"/>\n'
    y_ticks_svg += f'<text x="{margin_left - 8}" y="{y + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">{label}</text>\n'

# Generate SVG Bars & Points for 24 Hours
bars_svg = ""
pts_weekday = []
pts_weekend = []

for h in range(24):
    x = margin_left + (h / 23.0) * plot_w
    bar_w = 14
    
    # Weekday bar
    w_val = weekday_hours[h]
    bar_h = (w_val / max_val) * plot_h
    bar_y = 20 + plot_h - bar_h
    pts_weekday.append(f"{x:.1f},{bar_y:.1f}")
    
    # Weekend value
    we_val = weekend_hours[h]
    we_y = 20 + plot_h - (we_val / max_val) * plot_h
    pts_weekend.append(f"{x:.1f},{we_y:.1f}")
    
    # Bar rect
    color = "#38bdf8" if h not in [8, 18] else "#0284c7"
    if h == 18:
        color = "#38bdf8"
    bars_svg += f'<rect x="{x - bar_w/2:.1f}" y="{bar_y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" rx="3" opacity="0.85"><title>{h}시 평일: {w_val:,}건</title></rect>\n'
    
    # X axis label
    bars_svg += f'<text x="{x:.1f}" y="{20 + plot_h + 18}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="monospace">{h:02d}</text>\n'

# Polylines for curves
weekday_path_d = "M " + " L ".join(pts_weekday)
weekend_path_d = "M " + " L ".join(pts_weekend)

html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>대전 공영자전거 '타슈' 데이터 분석 및 AI 재배치 최적화 보고서</title>
    <style>
        :root {{
            --bg-body: #0b0f19;
            --bg-card: #111827;
            --bg-surface: #1e293b;
            --border: #243044;
            --border-focus: #38bdf8;
            --text-title: #f8fafc;
            --text-body: #cbd5e1;
            --text-muted: #64748b;
            --primary: #38bdf8;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --indigo: #818cf8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Pretendard", sans-serif;
            background-color: var(--bg-body);
            color: var(--text-body);
            line-height: 1.6;
            font-size: 14px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}
        
        /* Navigation */
        header {{
            position: sticky; top: 0; z-index: 50;
            background: rgba(11, 15, 25, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 14px 0;
        }}
        .nav-inner {{ display: flex; align-items: center; justify-content: space-between; }}
        .nav-title {{ font-size: 15px; font-weight: 700; color: var(--text-title); display: flex; align-items: center; gap: 8px; }}
        .nav-links {{ display: flex; gap: 14px; font-size: 13px; font-weight: 500; }}
        .nav-links a {{ color: var(--text-muted); text-decoration: none; transition: color 0.2s; }}
        .nav-links a:hover {{ color: var(--text-title); }}
        
        /* Section Titles */
        .section-header {{ margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }}
        .section-tag {{ font-size: 11px; font-weight: 700; color: var(--primary); text-transform: uppercase; letter-spacing: 0.5px; }}
        .section-title {{ font-size: 22px; font-weight: 800; color: var(--text-title); margin-top: 4px; }}
        .section-desc {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
        
        /* Cards & Grid */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
        @media (max-width: 860px) {{
            .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
        }}

        /* KPI Banner */
        .kpi-box {{ background: var(--bg-surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
        .kpi-label {{ font-size: 12px; color: var(--text-muted); font-weight: 600; }}
        .kpi-val {{ font-size: 26px; font-weight: 800; color: var(--text-title); font-family: monospace; margin: 4px 0; }}
        .kpi-sub {{ font-size: 11px; color: var(--text-muted); }}

        /* Tables */
        table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
        table.data-table th {{ padding: 10px 12px; background: var(--bg-surface); color: var(--text-muted); font-weight: 600; font-size: 12px; border-bottom: 1px solid var(--border); }}
        table.data-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
        table.data-table tr:hover td {{ background: rgba(30, 41, 59, 0.4); }}

        /* Tabs */
        .tab-btn {{
            padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: 600;
            background: var(--bg-surface); color: var(--text-muted); border: 1px solid var(--border);
            cursor: pointer; transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: rgba(56, 189, 248, 0.15); color: var(--primary); border-color: var(--primary);
        }}

        /* Callout */
        .callout {{
            background: rgba(30, 41, 59, 0.6);
            border-left: 4px solid var(--primary);
            padding: 14px 16px;
            border-radius: 0 8px 8px 0;
            margin: 14px 0;
            font-size: 13px;
        }}
        .badge {{
            display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;
        }}
        .badge-blue {{ background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3); }}
        .badge-green {{ background: rgba(52, 211, 153, 0.15); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.3); }}
        .badge-amber {{ background: rgba(251, 191, 36, 0.15); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.3); }}
        .badge-red {{ background: rgba(248, 113, 113, 0.15); color: var(--danger); border: 1px solid rgba(248, 113, 113, 0.3); }}
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="container nav-inner">
            <div class="nav-title">
                <span style="display:inline-block; width:10px; height:10px; background:var(--primary); border-radius:2px;"></span>
                <span>KT-X-AI | 대전 공영자전거 '타슈' AI 재배치 최적화 프로젝트</span>
                <span class="badge badge-blue">Branch: 성시준</span>
            </div>
            <nav class="nav-links">
                <a href="#sec-eda">1. 데이터 분석 결과</a>
                <a href="#sec-problem">2. 문제 정의 및 선행연구 한계</a>
                <a href="#sec-method">3. 제안 모델 및 AI 알고리즘</a>
                <a href="#sec-result">4. 실증 결과 및 개선 효과</a>
                <a href="#sec-synergy">5. 도시 인프라 연계</a>
            </nav>
        </div>
    </header>

    <main class="container" style="padding-top:28px; padding-bottom:60px;">

        <!-- Project Overview Summary Box -->
        <div class="card" style="border-left: 4px solid var(--primary); background: #0e1526;">
            <div style="font-size:12px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Executive Summary</div>
            <h1 style="font-size:22px; font-weight:800; color:var(--text-title); margin:6px 0 10px;">
                대전시 911만 건 통행 이력 기반 AI 수요 예측 및 물리 제약 차량 재배치 시스템
            </h1>
            <p style="font-size:13px; color:var(--text-body); line-height:1.7;">
                본 프로젝트는 대전광역시 공영자전거 '타슈'의 1개년 전수 통행 데이터(9,116,462건)와 1시간 단위 기상 관측 데이터를 결합하여 
                시민 통행 패턴을 정량 분석하고, 출퇴근 시간대 특정 대여소 품절(Deficit) 현상을 해결하기 위한 
                <strong>300m 생활권 Super-Station 군집화</strong>, <strong>Quantile 85% 비대칭 손실 XGBoost 모델</strong>, 
                <strong>물리 제약 반영 트럭 10대 동선 최적화</strong>, 그리고 <strong>포아송 베이지안 고장 감지 알고리즘</strong>을 구축·실증한 성과 보고서입니다.
            </p>

            <div class="grid-4" style="margin-top:20px;">
                <div class="kpi-box">
                    <div class="kpi-label">분석 통행 데이터</div>
                    <div class="kpi-val" style="color:var(--primary);">9,116,462건</div>
                    <div class="kpi-sub">1개년 1,200개 대여소 전수</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">생활권 Super-Station</div>
                    <div class="kpi-val" style="color:var(--success);">460개 앵커</div>
                    <div class="kpi-sub">300m 보행 반경 거점 집약</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">피크 결품 차단율</div>
                    <div class="kpi-val" style="color:var(--warning);">-65.0%</div>
                    <div class="kpi-sub">Top 20 결품률 38% ➔ 13.3%</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">트럭 10대 배차 완료</div>
                    <div class="kpi-val" style="color:var(--primary);">48.5분 / 176대</div>
                    <div class="kpi-sub">퇴근 60분 골든타임 이내 안착</div>
                </div>
            </div>
        </div>

        <!-- ========================================== -->
        <!-- 1. DATA ANALYSIS RESULTS (PUT FIRST AS REQUESTED) -->
        <!-- ========================================== -->
        <section id="sec-eda">
            <div class="section-header">
                <span class="section-tag">Section 01</span>
                <h2 class="section-title">1. 데이터 분석 결과</h2>
                <p class="section-desc">911만 건 전수 데이터로 도출한 시간대별 통행 리듬, 이동 거리·시간 분포, 기상 영향 및 핵심 통행 회랑(O-D Flow)</p>
            </div>

            <!-- 1-1. Hourly Pattern (Precise Graph) -->
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px;">
                    <div>
                        <h3 style="font-size:16px; font-weight:700; color:var(--text-title);">1-1. 시간대별 통행 패턴 분석: 평일 출퇴근 쌍봉형(M-Curve) vs 주말 여가형 분포</h3>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            평일 18:00 퇴근 피크 통행량(627,088건)은 08:00 출근 피크(393,736건) 대비 약 1.59배 높게 형성됨
                        </p>
                    </div>
                    <div style="display:flex; gap:16px; font-size:12px;">
                        <span style="display:flex; align-items:center; gap:6px;">
                            <span style="display:inline-block; width:12px; height:12px; background:var(--primary); border-radius:2px;"></span>
                            <span style="color:var(--text-title); font-weight:600;">평일 통행량 (바 차트)</span>
                        </span>
                        <span style="display:flex; align-items:center; gap:6px;">
                            <span style="display:inline-block; width:12px; height:3px; background:var(--success);"></span>
                            <span style="color:var(--text-title); font-weight:600;">주말 통행량 (추세선)</span>
                        </span>
                    </div>
                </div>

                <!-- SVG Hourly Chart (Clear, crisp, readable) -->
                <div style="width:100%; overflow-x:auto;">
                    <svg viewBox="0 0 {chart_w} {chart_h}" style="width:100%; min-width:800px; height:auto; background:#070c18; border-radius:8px; border:1px solid var(--border);">
                        <!-- Grid & Y-Ticks -->
                        {y_ticks_svg}

                        <!-- Weekend Line Chart -->
                        <path d="{weekend_path_d}" fill="none" stroke="var(--success)" stroke-width="2.5" stroke-dasharray="5 3"/>

                        <!-- Weekday Bars & X-Labels -->
                        {bars_svg}

                        <!-- Annotations for Peak Hours -->
                        <circle cx="{margin_left + (8/23.0)*plot_w}" cy="{20 + plot_h - (393736/max_val)*plot_h}" r="4" fill="#fff" stroke="#0284c7" stroke-width="2"/>
                        <text x="{margin_left + (8/23.0)*plot_w}" y="{20 + plot_h - (393736/max_val)*plot_h - 10}" fill="var(--text-title)" font-size="11" font-weight="700" text-anchor="middle">출근 08시 (39.4만)</text>

                        <circle cx="{margin_left + (18/23.0)*plot_w}" cy="{20 + plot_h - (627088/max_val)*plot_h}" r="4" fill="#fff" stroke="var(--danger)" stroke-width="2"/>
                        <text x="{margin_left + (18/23.0)*plot_w}" y="{20 + plot_h - (627088/max_val)*plot_h - 10}" fill="var(--danger)" font-size="11" font-weight="700" text-anchor="middle">퇴근 18시 (62.7만)</text>
                    </svg>
                </div>

                <div class="callout">
                    <strong style="color:var(--text-title);">분석 인사이트:</strong> 
                    평일에는 전형적인 출퇴근 통근형 M자 쌍봉이 나타나며, 특히 퇴근 시간대(17:00~19:00) 통행량이 전체의 18.1%를 차지합니다. 
                    출근 시간대에는 정시성을 위해 지하철·버스를 이용하는 반면, 퇴근 시간대에는 귀가, 약속, 운동 등이 결합되어 타슈 이용이 집중되므로 
                    <strong>오후 17:00 트럭 집중 재배치가 시스템 운영의 승패를 결정하는 골든타임</strong>임을 확인할 수 있습니다.
                </div>

                <!-- Exact 24 Hours Data Table -->
                <div style="margin-top:16px;">
                    <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">시간대별 상세 통행량 통계 (단위: 건)</div>
                    <div style="overflow-x:auto;">
                        <table class="data-table" style="font-family:monospace; font-size:11px;">
                            <thead>
                                <tr>
                                    <th>시간대</th>
                                    <th>00시</th><th>01시</th><th>02시</th><th>03시</th><th>04시</th><th>05시</th><th>06시</th><th>07시</th>
                                    <th style="color:var(--primary);">08시 (출근)</th><th>09시</th><th>10시</th><th>11시</th><th>12시</th><th>13시</th>
                                    <th>14시</th><th>15시</th><th>16시</th><th>17시</th><th style="color:var(--danger);">18시 (퇴근)</th>
                                    <th>19시</th><th>20시</th><th>21시</th><th>22시</th><th>23시</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style="font-weight:bold; color:var(--text-title);">평일</td>
                                    <td>12.1만</td><td>7.0만</td><td>4.3만</td><td>2.8만</td><td>2.8만</td><td>7.1만</td><td>13.1만</td><td>28.6만</td>
                                    <td style="color:var(--primary); font-weight:bold;">39.4만</td><td>24.0만</td><td>22.4만</td><td>27.6만</td><td>31.3만</td><td>29.7만</td>
                                    <td>31.6만</td><td>36.7만</td><td>45.2만</td><td>56.7만</td><td style="color:var(--danger); font-weight:bold;">62.7만</td>
                                    <td>45.6만</td><td>41.6만</td><td>39.0만</td><td>31.3만</td><td>24.2만</td>
                                </tr>
                                <tr>
                                    <td style="font-weight:bold; color:var(--success);">주말</td>
                                    <td>6.0만</td><td>3.8만</td><td>2.5만</td><td>1.7만</td><td>1.3만</td><td>2.1만</td><td>3.0만</td><td>4.7만</td>
                                    <td>7.5만</td><td>9.0만</td><td>10.4만</td><td>11.7만</td><td>13.7만</td><td>14.8만</td>
                                    <td>15.7만</td><td>16.8만</td><td>18.0만</td><td style="color:var(--success); font-weight:bold;">19.3만</td><td>18.0만</td>
                                    <td>15.9만</td><td>15.2만</td><td>14.1만</td><td>11.0만</td><td>9.0만</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- 1-2 & 1-3. Distance & Weather -->
            <div class="grid-2">
                <!-- 1-2. Distance and Duration -->
                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:4px;">1-2. 이동 거리 및 소요 시간 분포</h3>
                    <p style="font-size:12px; color:var(--text-muted); margin-bottom:16px;">중앙값 1.0 km / 11.0분 (라스트마일 마이크로 통행 특성)</p>

                    <!-- Horizontal Segmented Bar -->
                    <div style="height:22px; display:flex; border-radius:4px; overflow:hidden; font-size:11px; font-weight:700; line-height:22px; text-align:center;">
                        <div style="width:51.2%; background:var(--primary); color:#0b0f19;">51.2% (1km 이내)</div>
                        <div style="width:24.3%; background:var(--success); color:#0b0f19;">24.3% (1~2km)</div>
                        <div style="width:14.1%; background:var(--indigo); color:#fff;">14.1%</div>
                        <div style="width:10.4%; background:var(--warning); color:#0b0f19;">10.4%</div>
                    </div>

                    <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); margin-top:8px;">
                        <span>• 1km 이내: 51.2%</span>
                        <span>• 1~2km: 24.3% (누적 75.5%)</span>
                        <span>• 2~4km: 14.1%</span>
                        <span>• 4km 초과: 10.4%</span>
                    </div>

                    <div style="margin-top:16px; background:var(--bg-surface); padding:12px; border-radius:8px; font-size:12px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="color:var(--text-muted);">이동 거리 백분위수:</span>
                            <span>25% 0.5km | <strong>중앙값 1.0km</strong> | 75% 2.0km | 90% 4.2km</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:var(--text-muted);">소요 시간 백분위수:</span>
                            <span>25% 6.0분 | <strong>중앙값 11.0분</strong> | 75% 23.0분 | 90% 40.0분</span>
                        </div>
                    </div>

                    <p style="font-size:12px; color:var(--text-muted); margin-top:10px;">
                        전체 통행의 <strong>75.5%가 2km 이내 생활권에서 완결</strong>되며, 타슈는 도시 횡단 목적보다는 
                        지하철역·버스정류장에서 목적지까지의 도보 10~15분을 단축하는 라스트마일 수단으로 기능하고 있습니다.
                    </p>
                </div>

                <!-- 1-3. Weather Elasticity -->
                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:4px;">1-3. 기상 요인(강수량·기온)의 영향 분석</h3>
                    <p style="font-size:12px; color:var(--text-muted); margin-bottom:16px;">강수량 1mm 돌파 시 주간 통행량 70.9% 급감 (운영 임계선 확인)</p>

                    <table class="data-table" style="font-size:12px;">
                        <thead>
                            <tr>
                                <th>강수량 구간</th>
                                <th>시간당 통행량</th>
                                <th>변화율 (증감)</th>
                                <th>운영 권고 조치</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>비 안 옴 (0mm)</td>
                                <td style="font-family:monospace; font-weight:bold;">936.3 건/h</td>
                                <td>기준 (100%)</td>
                                <td><span class="badge badge-green">정상 운영</span> 트럭 10대 가동</td>
                            </tr>
                            <tr>
                                <td>0 ~ 1mm (이슬비)</td>
                                <td style="font-family:monospace;">650.1 건/h</td>
                                <td style="color:var(--primary);">-30.6%</td>
                                <td><span class="badge badge-blue">탄력 운영</span> 트럭 8대 가동</td>
                            </tr>
                            <tr style="background:rgba(248, 113, 113, 0.1);">
                                <td style="color:var(--danger); font-weight:bold;">1 ~ 3mm (일반 비)</td>
                                <td style="font-family:monospace; font-weight:bold; color:var(--danger);">272.5 건/h</td>
                                <td style="color:var(--danger); font-weight:bold;">-70.9% (임계점)</td>
                                <td><span class="badge badge-red">70% 감차</span> 트럭 3대만 대기</td>
                            </tr>
                            <tr>
                                <td>> 3mm (강한 비)</td>
                                <td style="font-family:monospace;">159.8 건/h</td>
                                <td style="color:var(--danger);">-82.9%</td>
                                <td><span class="badge badge-red">운행 중지</span> 침수지역 대피</td>
                            </tr>
                        </tbody>
                    </table>

                    <p style="font-size:12px; color:var(--text-muted); margin-top:12px;">
                        <strong>기온 영향:</strong> 18~30°C 구간이 시간당 약 1,100건으로 가장 활발하며, 
                        영하권(&lt;0°C)에서는 413건으로 61.3% 감소하여 혹서기보다 혹한기의 기피 현상이 더 뚜렷합니다.
                    </p>
                </div>
            </div>

            <!-- 1-4. O-D Flow Density & Zone-by-Zone Analysis -->
            <div class="card">
                <div style="margin-bottom:14px;">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title);">1-4. 주요 이동 경로(O-D Flow) 분석 및 3대 핵심 권역</h3>
                    <p style="font-size:12px; color:var(--text-muted);">
                        시 전체 통행 밀도 맵과 통행량이 가장 집중된 3대 권역(대학가 셔틀, 지하철 환승, 수목원 레저) 상세
                    </p>
                </div>

                <div class="grid-2">
                    <!-- Real Matplotlib O/D Density Heatmap Embedded -->
                    <div>
                        <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:6px;">대전시 전역 O/D 통행 밀도 지도 (GIS 실측 매핑)</div>
                        <div style="border-radius:8px; overflow:hidden; border:1px solid var(--border); background:#070c18; text-align:center;">
                            <img src="data:image/png;base64,{od_map_b64}" alt="대전시 O-D 통행 밀도 지도" style="width:100%; height:auto; display:block;">
                        </div>
                        <div style="font-size:11px; color:var(--text-muted); margin-top:6px;">
                            * 유성구(카이스트·충남대) 및 서구(둔산 정부청사·시청) 축선에 통행량의 65% 이상이 집중됨.
                        </div>
                    </div>

                    <!-- 3 Core Zones Detail -->
                    <div style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">3대 핵심 집중 권역 특성</div>
                            
                            <!-- Zone 1 -->
                            <div style="background:var(--bg-surface); padding:12px; border-radius:8px; margin-bottom:10px; border-left:3px solid var(--primary);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:var(--text-title); font-size:13px;">[권역 A] 유성·대학가 캠퍼스 셔틀 권역</strong>
                                    <span class="badge badge-blue">연간 약 210만 건</span>
                                </div>
                                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                    • 대표 경로: 카이스트 학사식당 ⇄ 창의학습관 (13,309건, 0.47km, 7.8분)<br/>
                                    • 특성: 강의동·기숙사·식당 간 평지 셔틀 이동. 양방향 통행 균형이 높아 자체 순환율 우수.
                                </div>
                            </div>

                            <!-- Zone 2 -->
                            <div style="background:var(--bg-surface); padding:12px; border-radius:8px; margin-bottom:10px; border-left:3px solid var(--success);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:var(--text-title); font-size:13px;">[권역 B] 둔산·정부청사 라스트마일 환승 권역</strong>
                                    <span class="badge badge-green">연간 약 145만 건</span>
                                </div>
                                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                    • 대표 경로: 정부청사역 4번출구 ➔ 정부청사 남문 (9,293건, 0.38km, 3.1분)<br/>
                                    • 특성: 지하철 1호선 하차 후 청사 내부로 통근하는 단방향 쏠림. 17:00 역방향 결품 급증.
                                </div>
                            </div>

                            <!-- Zone 3 -->
                            <div style="background:var(--bg-surface); padding:12px; border-radius:8px; border-left:3px solid var(--warning);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:var(--text-title); font-size:13px;">[권역 C] 만년·수목원·엑스포 수변 여가 권역</strong>
                                    <span class="badge badge-amber">연간 약 85만 건</span>
                                </div>
                                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                    • 대표 경로: 한밭수목원 서원 ⇄ 동원 (10,054건, 0.85km, 30.7분 체류)<br/>
                                    • 특성: 거리는 짧으나 체류시간이 30분 이상인 여가·산책 목적 통행. 주말 오후 집중.
                                </div>
                            </div>
                        </div>

                        <div style="font-size:11px; color:var(--text-muted); padding-top:10px;">
                            💡 결품이 발생하는 핵심 권역은 <strong>[권역 B] 둔산 행정타운</strong>이며, 자전거가 남아도는 잉여 권역은 <strong>노은·도안 등 주거지</strong>입니다.
                        </div>
                    </div>
                </div>

                <!-- Top 10 OD Table -->
                <div style="margin-top:16px;">
                    <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">대전시 최상위 O-D 통행 회랑 Top 8 통계</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>순위</th>
                                <th>출발 대여소</th>
                                <th>도착 대여소</th>
                                <th style="text-align:right;">연간 통행량</th>
                                <th style="text-align:right;">평균 거리</th>
                                <th style="text-align:right;">소요 시간</th>
                                <th>통행 유형</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="font-weight:bold; color:var(--primary);">1</td>
                                <td>어은동 카이스트 학사식당</td>
                                <td>구성동 카이스트 창의학습관</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">7,205 건</td>
                                <td style="text-align:right; font-family:monospace;">0.46 km</td>
                                <td style="text-align:right; font-family:monospace;">6.8 분</td>
                                <td>캠퍼스 셔틀</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--primary);">2</td>
                                <td>구성동 카이스트 창의학습관</td>
                                <td>어은동 카이스트 학사식당</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">6,104 건</td>
                                <td style="text-align:right; font-family:monospace;">0.49 km</td>
                                <td style="text-align:right; font-family:monospace;">8.7 분</td>
                                <td>캠퍼스 셔틀</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--primary);">3</td>
                                <td>어은동 카이스트 학사식당</td>
                                <td>구성동 카이스트 서쪽 쪽문</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">5,725 건</td>
                                <td style="text-align:right; font-family:monospace;">0.65 km</td>
                                <td style="text-align:right; font-family:monospace;">9.2 분</td>
                                <td>기숙사 통학</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--primary);">4</td>
                                <td>구성동 카이스트 창의학습관</td>
                                <td>구성동 카이스트 서쪽 쪽문</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">5,675 건</td>
                                <td style="text-align:right; font-family:monospace;">0.62 km</td>
                                <td style="text-align:right; font-family:monospace;">8.9 분</td>
                                <td>기숙사 통학</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--primary);">5</td>
                                <td>구성동 카이스트 정보전자동 (1410)</td>
                                <td>구성동 카이스트 서쪽 쪽문</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">5,153 건</td>
                                <td style="text-align:right; font-family:monospace;">0.72 km</td>
                                <td style="text-align:right; font-family:monospace;">7.5 분</td>
                                <td>연구실 이동</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--warning);">6</td>
                                <td>만년동 한밭수목원(서원)</td>
                                <td>만년동 한밭수목원(동원)</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">5,136 건</td>
                                <td style="text-align:right; font-family:monospace;">0.65 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--warning);">30.0 분</td>
                                <td>공원 여가</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--success);">7</td>
                                <td>둔산동 정부청사역 4번출구</td>
                                <td>둔산동 정부청사 입구(남문)</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">4,964 건</td>
                                <td style="text-align:right; font-family:monospace;">0.38 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--success);">3.1 분</td>
                                <td>지하철 환승</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--warning);">8</td>
                                <td>만년동 한밭수목원(동원)</td>
                                <td>만년동 한밭수목원(서원)</td>
                                <td style="text-align:right; font-family:monospace; font-weight:bold;">4,918 건</td>
                                <td style="text-align:right; font-family:monospace;">1.22 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--warning);">31.4 분</td>
                                <td>공원 여가</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- 2. PROBLEM STATEMENT & PRIOR WORK LIMITS -->
        <!-- ========================================== -->
        <section id="sec-problem" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 02</span>
                <h2 class="section-title">2. 문제 정의 및 선행 연구의 한계</h2>
                <p class="section-desc">현장 운영 병목과 기존 시계열 머신러닝·물류 최적화 방식의 구조적 결함</p>
            </div>

            <div class="grid-2">
                <!-- 2-1. Field Problems -->
                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:12px;">2-1. 현장 운영의 3대 핵심 문제</h3>
                    
                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--danger); font-size:13px;">① 출퇴근 비대칭 쏠림 및 품절(Stockout) 방치</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            퇴근 시간(17:00~19:00) 둔산동 정부청사, 시청, 지하철 역세권 20여 개 거점에서 대여 수요가 급증하여 
                            <strong>품절 방치율이 38.0%</strong>에 달함. 반면 노은동, 궁동 등 주거지에는 30~50대 이상 자전거가 과잉 누적(Surplus)됨.
                        </p>
                    </div>

                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--danger); font-size:13px;">② 트럭 10대의 비효율적 순회와 골든타임 상실</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            배차 트럭 10대가 예측 없이 민원 위주로 7~8개 대여소를 조금씩 순회하다가 
                            퇴근 피크(17:00~18:00) 60분의 골든타임을 놓치고 교통 체증에 갇혀 실질적 결품 구제 인원이 거점당 2명에 그침.
                        </p>
                    </div>

                    <div>
                        <strong style="color:var(--danger); font-size:13px;">③ 고장 유령 자전거(Ghost Bike)로 인한 시민 헛걸음</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            체인 이탈, 브레이크 고장 자전거가 대여소에 1대 남아있을 경우 공공 앱에는 '대여 가능 1대'로 표시되어 
                            시민들이 찾아왔다가 헛걸음하고 이탈하는 민원이 다수 발생함.
                        </p>
                    </div>
                </div>

                <!-- 2-2. Limitations of Prior Work -->
                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:12px;">2-2. 선행 연구 및 기존 접근의 3대 한계</h3>

                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--warning); font-size:13px;">① 1,200개 개별 대여소 모델링의 데이터 희소성(Sparsity)</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            기존 연구들은 1,200개 대여소를 각각 시계열로 예측함. 그러나 50~100m 간격으로 쪼개져 있어 
                            시간당 통행량이 0인 구간이 80%를 넘어 통계적 모델 학습이 무너짐.
                        </p>
                    </div>

                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--warning); font-size:13px;">② 평균제곱오차(MSE) 손실 함수의 치명적 결함</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            전통적 MSE는 +2대 잉여와 -2대 결품을 동일한 오차(4.0)로 취급함. 
                            그러나 모빌리티 서비스에서는 <strong>자전거가 모자란 것(결품)이 치명적</strong>이므로 평균값 회귀 시 피크 수요 버퍼를 확보하지 못함.
                        </p>
                    </div>

                    <div>
                        <strong style="color:var(--warning); font-size:13px;">③ 현장 물리 제약이 배제된 이상적 VRP 경로</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            기존 최적화 논문들은 트럭 주행 거리만 최소화할 뿐, <strong>정차 준비 3분, 상차 45초/대, 하차 30초/대</strong> 등 
                            현장 작업 소요시간과 퇴근 시내 정체(22km/h)를 반영하지 않아 현장에서 실행 불가능했음.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- 3. PROPOSED METHODOLOGY & AI INNOVATION -->
        <!-- ========================================== -->
        <section id="sec-method" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 03</span>
                <h2 class="section-title">3. 제안 모델 및 AI 알고리즘</h2>
                <p class="section-desc">300m Super-Station, Quantile 85% 비대칭 XGBoost, 물리 제약 차량 경로, 그리고 포아송 베이지안 고장 감지</p>
            </div>

            <div class="grid-3">
                <div class="card">
                    <span class="badge badge-blue">SPATIAL CONSOLIDATION</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">3-1. 300m 생활권 Super-Station</h3>
                    <p style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                        1,200개 대여소를 <strong>보행 3분(300m) 생활권 앵커 거점 460개</strong>로 집약 군집화(DBSCAN). 
                        시계열 데이터 밀도를 2.6배 개선하고 거점 간 도보 대체성을 확보함.
                    </p>
                </div>

                <div class="card">
                    <span class="badge badge-green">NETWORK DYNAMICS</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">3-2. O/D 87.4% 유입 회랑 피처</h3>
                    <p style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                        타 군집에서 출발하여 해당 거점으로 향하는 <strong>직전 시간 유입량(`lag1_ret`)</strong>을 네트워크 시계열 피처로 주입. 
                        XGBoost 모델 내 <strong>피처 중요도 53.41%</strong>를 차지하며 선제 예측력 확보.
                    </p>
                </div>

                <div class="card">
                    <span class="badge badge-amber">ASYMMETRIC ML</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">3-3. Quantile 85% (τ=0.85) Loss</h3>
                    <p style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                        MSE 대신 Pinball Loss(τ=0.85)를 적용하여 <strong>과소예측(부족 오차)에 5.67배 높은 페널티</strong>를 부과. 
                        출퇴근 피크 수요의 상위 85% 백분위수를 예측하여 안전 재고를 자동 확보.
                    </p>
                </div>
            </div>

            <div class="grid-2">
                <!-- Fleet Routing -->
                <div class="card">
                    <span class="badge badge-blue">LOGISTICS OPTIMIZATION</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">
                        3-4. 물리 제약 반영 트럭 10대 동선 최적화 (Fleet Routing)
                    </h3>
                    <ul style="font-size:12px; color:var(--text-muted); padding-left:18px; line-height:1.7;">
                        <li><strong>물리 제약 정량화:</strong> 정차 3분, 상차 45초/대, 하차 30초/대, 시내 주행 22km/h, 굴절률 1.3.</li>
                        <li><strong>[상차 1곳 ➔ 하차 1~2곳] 핀포인트 릴레이:</strong> 다중 순회를 배제하고 1-ton 트럭 1대당 잉여 거점 1곳에서 18대 만차 상차 후 인접 결품 거점 1~2곳에 전량 하차.</li>
                        <li><strong>결과:</strong> 평균 주행거리 7.5km, 총 작업시간 48.5분 만에 176대 하차 완료하여 60분 골든타임 이내 안착.</li>
                    </ul>
                </div>

                <!-- Bayesian Detector -->
                <div class="card">
                    <span class="badge badge-red">ONLINE FAULT INFERENCE</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">
                        3-5. 공공 API 스냅샷 기반 베이지안 고장 감지
                    </h3>
                    <div style="background:var(--bg-surface); padding:10px; border-radius:6px; font-family:monospace; font-size:11px; color:var(--warning); margin:6px 0;">
                        P(고장 | t시간 1대 고착) = 1 - exp(-λ × t)
                    </div>
                    <ul style="font-size:12px; color:var(--text-muted); padding-left:18px; line-height:1.7;">
                        <li><strong>원리:</strong> 시간당 대여 수요 λ인 거점에서 1대가 t시간 동안 대여되지 않을 확률 추론.</li>
                        <li><strong>성과:</strong> 시간당 수요 4대 이상인 거점에서 2시간 정체 시 <strong>사후 고장 확률 99.9% 확정</strong>.</li>
                        <li><strong>운영:</strong> 자전거 ID 없이도 유령 자전거를 특정하여 관제 유효 재고에서 자동 차감 및 수거 명령 발행.</li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- 4. FIELD RESULTS & OPERATIONAL IMPACT -->
        <!-- ========================================== -->
        <section id="sec-result" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 04</span>
                <h2 class="section-title">4. 실증 결과 및 개선 효과</h2>
                <p class="section-desc">2025년 9월 15일 17:00~18:00 퇴근 피크 1시간 실증 시뮬레이션 및 배차 결과</p>
            </div>

            <!-- Before vs After Metrics -->
            <div class="grid-3">
                <div class="card" style="text-align:center;">
                    <div style="font-size:12px; color:var(--text-muted);">Top 20 거점 결품 방치율</div>
                    <div style="font-size:32px; font-weight:900; color:var(--success); font-family:monospace; margin:6px 0;">
                        13.3% <span style="font-size:14px; color:var(--danger); text-decoration:line-through;">(38.0%)</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-title);">결품 위험 65.0% 급감</div>
                </div>

                <div class="card" style="text-align:center;">
                    <div style="font-size:12px; color:var(--text-muted);">퇴근 1시간 즉시 구제 인원</div>
                    <div style="font-size:32px; font-weight:900; color:var(--primary); font-family:monospace; margin:6px 0;">
                        73 명 <span style="font-size:14px; color:var(--text-muted);">/ 1시간</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-title);">핵심 거점 결품 방어율 88.8%</div>
                </div>

                <div class="card" style="text-align:center;">
                    <div style="font-size:12px; color:var(--text-muted);">트럭 10대 평균 작업 소요시간</div>
                    <div style="font-size:32px; font-weight:900; color:var(--warning); font-family:monospace; margin:6px 0;">
                        48.5 분 <span style="font-size:14px; color:var(--text-muted);">(7.5 km)</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-title);">총 176대 완벽 하차 완료</div>
                </div>
            </div>

            <!-- Dispatch Manifest Table -->
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div>
                        <h3 style="font-size:15px; font-weight:700; color:var(--text-title);">17:00 퇴근 피크 트럭 10대 실전 배차 지시서 (Dispatch Manifest)</h3>
                        <p style="font-size:11px; color:var(--text-muted);">1-ton 트럭 적재용량(18대), 상하차 실측 시간, 시내 주행 속도 제약 적용 결과</p>
                    </div>
                    <span class="badge badge-green">10대 전원 60분 이내 완료</span>
                </div>

                <table class="data-table">
                    <thead>
                        <tr>
                            <th>차량</th>
                            <th>상차 거점 (잉여 회수)</th>
                            <th>하차 거점 (결품 투하)</th>
                            <th style="text-align:center;">배차 수량</th>
                            <th style="text-align:right;">주행 거리</th>
                            <th style="text-align:right;">총 소요시간</th>
                            <th>도착 상태</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="font-weight:bold; color:var(--primary);">트럭 01</td>
                            <td>노은역 3번출구 (잉여 45대)</td>
                            <td>유성온천역 7번출구 (결품 18대)</td>
                            <td style="text-align:center; font-weight:bold; color:var(--success);">18 대</td>
                            <td style="text-align:right; font-family:monospace;">6.8 km</td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">46.2 분</td>
                            <td><span class="badge badge-green">완료 (17:46)</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--primary);">트럭 02</td>
                            <td>궁동 로데오거리 (잉여 38대)</td>
                            <td>월평역 1번출구 (결품 16대)</td>
                            <td style="text-align:center; font-weight:bold; color:var(--success);">16 대</td>
                            <td style="text-align:right; font-family:monospace;">5.4 km</td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">42.8 분</td>
                            <td><span class="badge badge-green">완료 (17:42)</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--primary);">트럭 03</td>
                            <td>만년동 KBS 방송국 (잉여 32대)</td>
                            <td>정부청사역 2번출구 (결품 18대)</td>
                            <td style="text-align:center; font-weight:bold; color:var(--success);">18 대</td>
                            <td style="text-align:right; font-family:monospace;">4.9 km</td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">41.5 분</td>
                            <td><span class="badge badge-green">완료 (17:41)</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--primary);">트럭 04</td>
                            <td>도안 수변공원 (잉여 29대)</td>
                            <td>원신흥동 주민센터 (결품 18대)</td>
                            <td style="text-align:center; font-weight:bold; color:var(--success);">18 대</td>
                            <td style="text-align:right; font-family:monospace;">7.2 km</td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">47.8 분</td>
                            <td><span class="badge badge-green">완료 (17:47)</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--primary);">트럭 05</td>
                            <td>탄방동 남선공원 (잉여 27대)</td>
                            <td>시청역 8번출구 (결품 18대)</td>
                            <td style="text-align:center; font-weight:bold; color:var(--success);">18 대</td>
                            <td style="text-align:right; font-family:monospace;">6.1 km</td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">45.0 분</td>
                            <td><span class="badge badge-green">완료 (17:45)</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--primary);">트럭 06~10</td>
                            <td>관저동/가오동/송촌동 잉여 거점</td>
                            <td>대동역/서대전네거리역 결품 거점</td>
                            <td style="text-align:center; font-weight:bold; color:var(--success);">88 대</td>
                            <td style="text-align:right; font-family:monospace;">평균 8.8 km</td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">평균 51.4 분</td>
                            <td><span class="badge badge-green">완료 (17:51)</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- 5. URBAN MOBILITY EXPANSION & SYNERGY -->
        <!-- ========================================== -->
        <section id="sec-synergy" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 05</span>
                <h2 class="section-title">5. 도시 인프라 확장 및 정책 연계</h2>
                <p class="section-desc">2028년 트램 2호선 연계, 지하철 1호선 환승 편익, 자전거 도로 안전성 개선, 그리고 ESG 탄소 감축</p>
            </div>

            <div class="grid-4">
                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--primary);">TRAM LINE 2 SYNERGY</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">대전 트램 2호선 연계</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--primary); font-family:monospace;">42.4%</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        2028년 개통 예정인 45개 트램 정거장 반경 500m 이내에 <strong>전체 타슈 통행의 42.4%(386.3만 건)</strong>가 집중되어 최적의 피더(Feeder) 모빌리티 역할을 수행함.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--success);">METRO 1 FEEDER</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">지하철 1호선 환승</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--success); font-family:monospace;">19.4%</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        22개 지하철역 반경 300m 이내에서 연간 <strong>176.6만 건</strong>의 퍼스트·라스트마일 환승 통행 발생 증빙.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--warning);">ROAD SAFETY</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">자전거도로 단절 Top 10</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--warning); font-family:monospace;">Top 10</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        OSM 도로망 오버레이로 월 10만 건 이상 통행되나 전용 도로가 끊겨 차도로 내몰리는 위험 구간 1순위 도로 투자처 제시.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--danger);">ESG BENEFIT</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">ESG 탄소 순감축량</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--danger); font-family:monospace;">701.8 t</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        연간 701.8톤 CO₂ 감축(소나무 10.6만 그루 식재 효과) 및 시민 유류비 6.89억 원 절감 공인 정량화.
                    </p>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer style="border-top:1px solid var(--border); padding:28px 0; text-align:center; font-size:12px; color:var(--text-muted); background:#070c18;">
        <div class="container">
            <div style="color:var(--text-body); font-weight:600; margin-bottom:4px;">
                KT-X-AI 대전 공영자전거 '타슈' 지능형 모빌리티 솔루션
            </div>
            <p>프로젝트 총괄: 성시준 (Sijun Sung) | 페어 프로그래밍 AI: Google Deepmind Antigravity</p>
            <p style="margin-top:2px;">저장소: <a href="https://github.com/sijoon-sung/KT-X-AI-/tree/성시준" target="_blank" style="color:var(--primary); text-decoration:none;">github.com/sijoon-sung/KT-X-AI- (Branch: 성시준)</a></p>
        </div>
    </footer>

</body>
</html>
"""

# Destination paths
dest_root = ROOT / "index.html"
dest_docs = ROOT / "docs/index.html"
dest_desktop = Path("C:/Users/sijoo/OneDrive/바탕 화면/타슈_재배치_최적화_최종결과패키지/00_타슈_AI_프로젝트_종합_웹포트폴리오.html")
dest_brain = Path("C:/Users/sijoo/.gemini/antigravity/brain/69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c/tashu_project_portfolio.html")

os.makedirs(ROOT / "docs", exist_ok=True)

with open(dest_root, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Saved: {dest_root}")

with open(dest_docs, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Saved: {dest_docs}")

with open(dest_desktop, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Saved: {dest_desktop}")

with open(dest_brain, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Saved: {dest_brain}")
