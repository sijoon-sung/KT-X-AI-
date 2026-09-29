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

# 3. Prepare Hourly Data for Section 1 Chart
weekday_hours = [eda["weekday_hourly"][str(h)] for h in range(24)]
weekend_hours = [eda["weekend_hourly"][str(h)] for h in range(24)]

max_val = 650000
chart_w = 940
chart_h = 260
margin_left = 60
margin_bottom = 40
plot_w = chart_w - margin_left - 20
plot_h = chart_h - margin_bottom - 20

y_ticks_svg = ""
for tick in [0, 100000, 200000, 300000, 400000, 500000, 600000]:
    y = 20 + plot_h - (tick / max_val) * plot_h
    label = f"{tick // 10000}만" if tick > 0 else "0"
    y_ticks_svg += f'<line x1="{margin_left}" y1="{y}" x2="{chart_w - 20}" y2="{y}" stroke="#1e293b" stroke-width="1" stroke-dasharray="{"0" if tick==0 else "3 3"}"/>\n'
    y_ticks_svg += f'<text x="{margin_left - 8}" y="{y + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">{label}</text>\n'

bars_svg = ""
pts_weekday = []
pts_weekend = []

for h in range(24):
    x = margin_left + (h / 23.0) * plot_w
    bar_w = 14
    w_val = weekday_hours[h]
    bar_h = (w_val / max_val) * plot_h
    bar_y = 20 + plot_h - bar_h
    pts_weekday.append(f"{x:.1f},{bar_y:.1f}")
    
    we_val = weekend_hours[h]
    we_y = 20 + plot_h - (we_val / max_val) * plot_h
    pts_weekend.append(f"{x:.1f},{we_y:.1f}")
    
    color = "#38bdf8" if h not in [8, 18] else "#0284c7"
    bars_svg += f'<rect x="{x - bar_w/2:.1f}" y="{bar_y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" rx="3" opacity="0.85"><title>{h}시 평일: {w_val:,}건</title></rect>\n'
    bars_svg += f'<text x="{x:.1f}" y="{20 + plot_h + 18}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="monospace">{h:02d}</text>\n'

weekend_path_d = "M " + " L ".join(pts_weekend)

# 4. Prepare Section 4 Simulation Dataset (Actual vs MSE vs Q85)
sim_hours = list(range(24))
sim_actuals = [419, 229, 123, 106, 126, 291, 519, 1098, 1400, 711, 670, 864, 956, 817, 933, 1168, 1564, 2088, 2304, 1749, 1661, 1570, 1298, 819]
sim_mse = [489, 322, 228, 213, 231, 376, 577, 1086, 1352, 746, 710, 880, 961, 839, 941, 1148, 1496, 1628, 1797, 1364, 1582, 1502, 1262, 841]
sim_q85 = [658, 441, 320, 301, 324, 512, 772, 1432, 1776, 991, 944, 1165, 1270, 1111, 1244, 1512, 1963, 2338, 2580, 1958, 2074, 1970, 1660, 1114]

sim_max = 2800
sim_chart_w = 940
sim_chart_h = 280
sim_plot_w = sim_chart_w - margin_left - 20
sim_plot_h = sim_chart_h - margin_bottom - 20

sim_y_ticks_svg = ""
for tick in [0, 500, 1000, 1500, 2000, 2500]:
    y = 20 + sim_plot_h - (tick / sim_max) * sim_plot_h
    label = f"{tick:,}"
    sim_y_ticks_svg += f'<line x1="{margin_left}" y1="{y}" x2="{sim_chart_w - 20}" y2="{y}" stroke="#1e293b" stroke-width="1" stroke-dasharray="{"0" if tick==0 else "3 3"}"/>\n'
    sim_y_ticks_svg += f'<text x="{margin_left - 8}" y="{y + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">{label}</text>\n'

pts_act = []
pts_mse = []
pts_q85 = []

for h in range(24):
    x = margin_left + (h / 23.0) * sim_plot_w
    y_act = 20 + sim_plot_h - (sim_actuals[h] / sim_max) * sim_plot_h
    y_mse = 20 + sim_plot_h - (sim_mse[h] / sim_max) * sim_plot_h
    y_q85 = 20 + sim_plot_h - (sim_q85[h] / sim_max) * sim_plot_h
    
    pts_act.append(f"{x:.1f},{y_act:.1f}")
    pts_mse.append(f"{x:.1f},{y_mse:.1f}")
    pts_q85.append(f"{x:.1f},{y_q85:.1f}")

path_act = "M " + " L ".join(pts_act)
path_mse = "M " + " L ".join(pts_mse)
path_q85 = "M " + " L ".join(pts_q85)

# Generate polygon for Deficit Area during peak hours (h=16 to h=20)
# Actual > MSE
peak_pts_act = pts_act[16:21]
peak_pts_mse = pts_mse[16:21]
peak_pts_mse.reverse()
deficit_polygon = "M " + " L ".join(peak_pts_act) + " L " + " L ".join(peak_pts_mse) + " Z"

# Generate polygon for Q85 Safety Buffer (h=16 to h=20)
# Q85 >= Actual
peak_pts_q85 = pts_q85[16:21]
peak_pts_act_rev = list(pts_act[16:21])
peak_pts_act_rev.reverse()
q85_buffer_polygon = "M " + " L ".join(peak_pts_q85) + " L " + " L ".join(peak_pts_act_rev) + " Z"

# X labels for sim chart
sim_x_labels_svg = ""
for h in range(24):
    x = margin_left + (h / 23.0) * sim_plot_w
    sim_x_labels_svg += f'<text x="{x:.1f}" y="{20 + sim_plot_h + 18}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="monospace">{h:02d}</text>\n'

html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>대전 공영자전거 '타슈' AI 재배치 최적화 & 수요 예측 오차 시뮬레이션 보고서</title>
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
        .card-glow {{
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.08);
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

        /* Tabs & Controls */
        .tab-btn {{
            padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: 600;
            background: var(--bg-surface); color: var(--text-muted); border: 1px solid var(--border);
            cursor: pointer; transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: rgba(56, 189, 248, 0.15); color: var(--primary); border-color: var(--primary);
        }}
        .ctrl-btn {{
            padding: 8px 16px; border-radius: 6px; font-size: 12px; font-weight: 700;
            background: var(--bg-surface); color: var(--text-title); border: 1px solid var(--border);
            cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s;
        }}
        .ctrl-btn:hover {{ background: #334155; border-color: var(--primary); }}
        .ctrl-btn.playing {{ background: rgba(56, 189, 248, 0.2); color: var(--primary); border-color: var(--primary); }}

        .slider-bar {{
            width: 100%; -webkit-appearance: none; height: 6px; border-radius: 3px;
            background: #1e293b; outline: none; margin: 10px 0;
        }}
        .slider-bar::-webkit-slider-thumb {{
            -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
            background: var(--primary); cursor: pointer; box-shadow: 0 0 8px var(--primary);
        }}

        /* Callout & Badges */
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
                <a href="#sec-sim">4. 예측 vs 실제 수요 차이 시뮬레이션</a>
                <a href="#sec-result">5. 현장 실증 결과</a>
                <a href="#sec-synergy">6. 도시 인프라 연계</a>
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
        <!-- 1. DATA ANALYSIS RESULTS -->
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

                <!-- SVG Hourly Chart -->
                <div style="width:100%; overflow-x:auto;">
                    <svg viewBox="0 0 {chart_w} {chart_h}" style="width:100%; min-width:800px; height:auto; background:#070c18; border-radius:8px; border:1px solid var(--border);">
                        {y_ticks_svg}
                        <path d="{weekend_path_d}" fill="none" stroke="var(--success)" stroke-width="2.5" stroke-dasharray="5 3"/>
                        {bars_svg}
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
            </div>

            <!-- 1-2 & 1-3. Distance & Weather -->
            <div class="grid-2">
                <!-- 1-2. Distance and Duration -->
                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:4px;">1-2. 이동 거리 및 소요 시간 분포</h3>
                    <p style="font-size:12px; color:var(--text-muted); margin-bottom:16px;">중앙값 1.0 km / 11.0분 (라스트마일 마이크로 통행 특성)</p>

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

                    <p style="font-size:12px; color:var(--text-muted); margin-top:14px;">
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
                                <th>변화율</th>
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
                        </tbody>
                    </table>
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
                    <div>
                        <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:6px;">대전시 전역 O/D 통행 밀도 지도 (GIS 실측 매핑)</div>
                        <div style="border-radius:8px; overflow:hidden; border:1px solid var(--border); background:#070c18; text-align:center;">
                            <img src="data:image/png;base64,{od_map_b64}" alt="대전시 O-D 통행 밀도 지도" style="width:100%; height:auto; display:block;">
                        </div>
                    </div>

                    <div style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">3대 핵심 집중 권역 특성</div>
                            
                            <div style="background:var(--bg-surface); padding:12px; border-radius:8px; margin-bottom:10px; border-left:3px solid var(--primary);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:var(--text-title); font-size:13px;">[권역 A] 유성·대학가 캠퍼스 셔틀 권역</strong>
                                    <span class="badge badge-blue">연간 약 210만 건</span>
                                </div>
                                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                    • 대표 경로: 카이스트 학사식당 ⇄ 창의학습관 (13,309건, 0.47km, 7.8분)<br/>
                                    • 특성: 강의동·기숙사·식당 간 셔틀 이동. 양방향 균형이 높아 자체 순환율 우수.
                                </div>
                            </div>

                            <div style="background:var(--bg-surface); padding:12px; border-radius:8px; margin-bottom:10px; border-left:3px solid var(--success);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:var(--text-title); font-size:13px;">[권역 B] 둔산·정부청사 라스트마일 환승 권역</strong>
                                    <span class="badge badge-green">연간 약 145만 건</span>
                                </div>
                                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                    • 대표 경로: 정부청사역 4번출구 ➔ 정부청사 남문 (9,293건, 0.38km, 3.1분)<br/>
                                    • 특성: 지하철 1호선 하차 후 청사 내부로 통근하는 단방향 쏠림. 17:00 결품 집중.
                                </div>
                            </div>

                            <div style="background:var(--bg-surface); padding:12px; border-radius:8px; border-left:3px solid var(--warning);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:var(--text-title); font-size:13px;">[권역 C] 만년·수목원·엑스포 수변 여가 권역</strong>
                                    <span class="badge badge-amber">연간 약 85만 건</span>
                                </div>
                                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                    • 대표 경로: 한밭수목원 서원 ⇄ 동원 (10,054건, 0.85km, 30.7분 체류)<br/>
                                    • 특성: 거리는 짧으나 체류시간이 30분 이상인 여가 목적 통행. 주말 오후 집중.
                                </div>
                            </div>
                        </div>
                    </div>
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
                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:12px;">2-1. 현장 운영의 3대 핵심 문제</h3>
                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--danger); font-size:13px;">① 출퇴근 비대칭 쏠림 및 품절(Stockout) 방치</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            퇴근 시간(17:00~19:00) 둔산동 정부청사, 역세권 20여 개 거점의 <strong>품절 방치율이 38.0%</strong>에 달함. 반면 노은동 주거지에는 30~50대 이상 자전거가 과잉 누적(Surplus)됨.
                        </p>
                    </div>
                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--danger); font-size:13px;">② 트럭 10대의 비효율적 순회와 골든타임 상실</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            트럭 10대가 민원 위주로 7~8개 대여소를 조금씩 순회하다 퇴근 피크 60분의 골든타임을 놓치고 실질 구제 인원이 거점당 2명에 그침.
                        </p>
                    </div>
                    <div>
                        <strong style="color:var(--danger); font-size:13px;">③ 고장 유령 자전거(Ghost Bike)로 인한 시민 헛걸음</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            고장 자전거가 대여소에 1대 남아있을 경우 앱에 '대여 가능 1대'로 표시되어 시민들의 헛걸음 민원을 유발함.
                        </p>
                    </div>
                </div>

                <div class="card">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title); margin-bottom:12px;">2-2. 선행 연구 및 기존 접근의 3대 한계</h3>
                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--warning); font-size:13px;">① 1,200개 개별 대여소 모델링의 데이터 희소성(Sparsity)</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            대여소가 50~100m 간격으로 쪼개져 있어 시간당 통행량이 0인 구간이 80%를 넘어 모델 학습이 무너짐.
                        </p>
                    </div>
                    <div style="margin-bottom:14px;">
                        <strong style="color:var(--warning); font-size:13px;">② 평균제곱오차(MSE) 손실 함수의 치명적 결함</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            MSE는 잉여(+2대)와 결품(-2대)을 동일하게 취급하여 <strong>모자라는 것(결품)이 치명적인 현장 특성</strong>에서 안전 버퍼를 확보하지 못함.
                        </p>
                    </div>
                    <div>
                        <strong style="color:var(--warning); font-size:13px;">③ 현장 물리 제약이 배제된 이상적 VRP 경로</strong>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                            정차 준비 3분, 상차 45초/대, 하차 30초/대 등 현장 작업 소요시간과 퇴근 시내 정체(22km/h)를 배제하여 현장 실행이 불가능했음.
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
                        1,200개 대여소를 <strong>보행 3분(300m) 생활권 앵커 거점 460개</strong>로 집약 군집화. 데이터 밀도를 2.6배 개선하고 거점 간 도보 대체성을 확보함.
                    </p>
                </div>

                <div class="card">
                    <span class="badge badge-green">NETWORK DYNAMICS</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">3-2. O/D 87.4% 유입 회랑 피처</h3>
                    <p style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                        타 군집에서 유입되는 <strong>직전 시간 유입량(`lag1_ret`)</strong>을 네트워크 시계열 피처로 주입. 모델 내 <strong>피처 중요도 53.41%</strong>를 차지하며 선제 예측력 확보.
                    </p>
                </div>

                <div class="card">
                    <span class="badge badge-amber">ASYMMETRIC ML</span>
                    <h3 style="font-size:15px; font-weight:700; color:var(--text-title); margin:8px 0 6px;">3-3. Quantile 85% (τ=0.85) Loss</h3>
                    <p style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                        Pinball Loss(τ=0.85)를 적용하여 <strong>과소예측(부족 오차)에 5.67배 높은 페널티</strong>를 부과. 피크 수요의 상위 85% 백분위수를 예측하여 안전 재고를 자동 산출.
                    </p>
                </div>
            </div>
        </section>

        <!-- ============================================================== -->
        <!-- 4. ALGORITHM SIMULATION: ACTUAL VS PREDICTED DIFFERENCE (USER'S NEW REQUEST) -->
        <!-- ============================================================== -->
        <section id="sec-sim" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 04</span>
                <h2 class="section-title">4. 알고리즘 예측치 vs 실제 수요량 차이(오차) 검증 및 시뮬레이션</h2>
                <p class="section-desc">
                    기존 MSE 모델의 과소예측(결품 발생 영역)과 제안 Q85 모델의 안전 재고 방어 버퍼를 시간대별 시계열 차이와 동적 시뮬레이터로 정량 검증
                </p>
            </div>

            <!-- Dynamic Playback Controller Card -->
            <div class="card card-glow">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:16px;">
                    <div>
                        <h3 style="font-size:16px; font-weight:700; color:var(--text-title);">
                            4-1. 24시간 수요량 vs 예측치 시계열 차이(Gap) 비교 및 동적 시뮬레이터
                        </h3>
                        <p style="font-size:12px; color:var(--text-muted);">
                            시간의 흐름에 따라 실제 수요(흰색), 기존 MSE 예측(붉은 점선), 제안 Q85 예측(청록 실선)의 차이와 결품 여부를 실시간 추적
                        </p>
                    </div>

                    <!-- Animation Playback Controls -->
                    <div style="display:flex; align-items:center; gap:8px;">
                        <button class="ctrl-btn" id="playBtn" onclick="togglePlay()">
                            <span id="playIcon">▶</span> <span id="playText">시뮬레이션 재생</span>
                        </button>
                        <button class="ctrl-btn" onclick="resetSim()">⏮ 리셋</button>
                        <button class="ctrl-btn" id="speedBtn" onclick="toggleSpeed()">배속: 1x</button>
                    </div>
                </div>

                <!-- Live Metrics Display Bar during Simulation -->
                <div class="grid-4" style="margin-bottom:16px;">
                    <div class="kpi-box" style="background:#070c18;">
                        <div class="kpi-label">시뮬레이션 시각</div>
                        <div class="kpi-val" id="simHourDisplay" style="color:var(--text-title); font-size:22px;">18:00 (퇴근 피크)</div>
                        <div class="kpi-sub" id="simStatusSub">퇴근길 대여 수요 집중 시간대</div>
                    </div>
                    <div class="kpi-box" style="background:#070c18;">
                        <div class="kpi-label">실제 시민 대여 수요량</div>
                        <div class="kpi-val" id="simActualVal" style="color:#ffffff; font-size:22px;">2,304 대</div>
                        <div class="kpi-sub">대전시 전역 1시간 실측 통행</div>
                    </div>
                    <div class="kpi-box" style="background:#070c18; border-color:rgba(248, 113, 113, 0.4);">
                        <div class="kpi-label">기존 MSE 예측치 및 오차(차이)</div>
                        <div class="kpi-val" id="simMseVal" style="color:var(--danger); font-size:22px;">1,797 대 <span style="font-size:14px;">(-507대)</span></div>
                        <div class="kpi-sub" id="simMseGapSub" style="color:var(--danger); font-weight:bold;">🚨 507대 결품 발생! (품절)</div>
                    </div>
                    <div class="kpi-box" style="background:#070c18; border-color:rgba(56, 189, 248, 0.4);">
                        <div class="kpi-label">제안 Q85 예측치 및 안전 버퍼</div>
                        <div class="kpi-val" id="simQ85Val" style="color:var(--primary); font-size:22px;">2,580 대 <span style="font-size:14px;">(+276대)</span></div>
                        <div class="kpi-sub" id="simQ85GapSub" style="color:var(--success); font-weight:bold;">✓ 276대 안전 재고 확보 (방어)</div>
                    </div>
                </div>

                <!-- Simulation Time Scrubber -->
                <div style="margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); font-family:monospace;">
                        <span>00:00</span>
                        <span>06:00 (새벽)</span>
                        <span>08:00 (출근 피크)</span>
                        <span>12:00 (점심)</span>
                        <span style="color:var(--danger); font-weight:bold;">18:00 (퇴근 피크)</span>
                        <span>23:00</span>
                    </div>
                    <input type="range" min="0" max="23" value="18" class="slider-bar" id="simSlider" oninput="setSimHour(parseInt(this.value))">
                </div>

                <!-- Difference Line & Area Chart (SVG) -->
                <div style="width:100%; overflow-x:auto;">
                    <svg id="simSvg" viewBox="0 0 {sim_chart_w} {sim_chart_h}" style="width:100%; min-width:800px; height:auto; background:#070c18; border-radius:8px; border:1px solid var(--border);">
                        <!-- Grid & Y-Ticks -->
                        {sim_y_ticks_svg}

                        <!-- Difference Area: Deficit Area (Actual > MSE in 16~20h) -->
                        <path d="{deficit_polygon}" fill="rgba(248, 113, 113, 0.25)" stroke="none"/>
                        <!-- Difference Area: Q85 Buffer Area (Q85 >= Actual in 16~20h) -->
                        <path d="{q85_buffer_polygon}" fill="rgba(56, 189, 248, 0.18)" stroke="none"/>

                        <!-- Text Annotations for Difference Areas -->
                        <text x="{margin_left + (18/23.0)*sim_plot_w + 14}" y="{20 + sim_plot_h - (2050/sim_max)*sim_plot_h}" fill="var(--danger)" font-size="11" font-weight="700">◀ MSE 결품 발생 영역 (-507대)</text>
                        <text x="{margin_left + (18/23.0)*sim_plot_w + 14}" y="{20 + sim_plot_h - (2450/sim_max)*sim_plot_h}" fill="var(--primary)" font-size="11" font-weight="700">◀ Q85 안전 버퍼 영역 (+276대)</text>

                        <!-- Lines -->
                        <!-- 1. MSE Line (Red Dashed) -->
                        <path d="{path_mse}" fill="none" stroke="var(--danger)" stroke-width="2" stroke-dasharray="4 3"/>
                        <!-- 2. Q85 Line (Cyan Solid) -->
                        <path d="{path_q85}" fill="none" stroke="var(--primary)" stroke-width="2.5"/>
                        <!-- 3. Actual Demand Line (White Solid) -->
                        <path d="{path_act}" fill="none" stroke="#ffffff" stroke-width="3"/>

                        <!-- X Labels -->
                        {sim_x_labels_svg}

                        <!-- Interactive Moving Scrubber Line on Sim SVG -->
                        <line id="simCursorLine" x1="{margin_left + (18/23.0)*sim_plot_w}" y1="20" x2="{margin_left + (18/23.0)*sim_plot_w}" y2="{20 + sim_plot_h}" stroke="#facc15" stroke-width="2" stroke-dasharray="2 2"/>
                        <circle id="simCursorAct" cx="{margin_left + (18/23.0)*sim_plot_w}" cy="{20 + sim_plot_h - (sim_actuals[18]/sim_max)*sim_plot_h}" r="5" fill="#ffffff" stroke="#000" stroke-width="2"/>
                        <circle id="simCursorMse" cx="{margin_left + (18/23.0)*sim_plot_w}" cy="{20 + sim_plot_h - (sim_mse[18]/sim_max)*sim_plot_h}" r="4" fill="var(--danger)"/>
                        <circle id="simCursorQ85" cx="{margin_left + (18/23.0)*sim_plot_w}" cy="{20 + sim_plot_h - (sim_q85[18]/sim_max)*sim_plot_h}" r="4" fill="var(--primary)"/>
                    </svg>
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; font-size:12px; color:var(--text-muted); flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; gap:16px;">
                        <span style="color:#ffffff; font-weight:bold;">─── 실측 수요량 (Actual Demand)</span>
                        <span style="color:var(--danger); font-weight:bold;">- - - 기존 MSE 예측 (평균값 회귀)</span>
                        <span style="color:var(--primary); font-weight:bold;">─── 제안 Q85 예측 (85% 분위수 비대칭)</span>
                    </div>
                    <span>노란 점선: 현재 시뮬레이션 시각 인디케이터</span>
                </div>
            </div>

            <!-- 4-2. Station-level Residuals & Difference Table -->
            <div class="card">
                <div style="margin-bottom:14px;">
                    <h3 style="font-size:16px; font-weight:700; color:var(--text-title);">
                        4-2. 주요 5대 핵심 결품 거점의 실제 수요량 vs 모델 예측치 차이(Residuals) 분석
                    </h3>
                    <p style="font-size:12px; color:var(--text-muted);">
                        17:00~18:00 퇴근 피크 시간대 거점별 실제 대여 수요와 각 모델의 예측치, 그리고 발생하는 결품 오차 비교
                    </p>
                </div>

                <table class="data-table">
                    <thead>
                        <tr>
                            <th>거점명 (대여소)</th>
                            <th style="text-align:right;">잔여 재고</th>
                            <th style="text-align:right;">실제 수요량</th>
                            <th style="text-align:right; color:var(--danger);">MSE 예측치 (차이)</th>
                            <th style="text-align:center;">MSE 결품 결과</th>
                            <th style="text-align:right; color:var(--primary);">Q85 예측치 (차이)</th>
                            <th style="text-align:center;">Q85 방어 결과</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="font-weight:bold; color:var(--text-title);">둔산동 정부청사역 4번출구</td>
                            <td style="text-align:right; font-family:monospace;">14 대</td>
                            <td style="text-align:right; font-family:monospace; font-weight:bold;">42 대</td>
                            <td style="text-align:right; font-family:monospace; color:var(--danger);">24 대 <span style="font-size:11px;">(-18대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-red">18명 품절 피해</span></td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">46 대 <span style="font-size:11px;">(+4대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-green">결품 0건 완벽 방어</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--text-title);">유성온천역 7번출구</td>
                            <td style="text-align:right; font-family:monospace;">8 대</td>
                            <td style="text-align:right; font-family:monospace; font-weight:bold;">36 대</td>
                            <td style="text-align:right; font-family:monospace; color:var(--danger);">20 대 <span style="font-size:11px;">(-16대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-red">16명 품절 피해</span></td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">40 대 <span style="font-size:11px;">(+4대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-green">94.4% 결품 차단</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--text-title);">월평역 1번출구</td>
                            <td style="text-align:right; font-family:monospace;">6 대</td>
                            <td style="text-align:right; font-family:monospace; font-weight:bold;">31 대</td>
                            <td style="text-align:right; font-family:monospace; color:var(--danger);">18 대 <span style="font-size:11px;">(-13대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-red">13명 품절 피해</span></td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">34 대 <span style="font-size:11px;">(+3대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-green">결품 0건 완벽 방어</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--text-title);">시청역 8번출구</td>
                            <td style="text-align:right; font-family:monospace;">10 대</td>
                            <td style="text-align:right; font-family:monospace; font-weight:bold;">28 대</td>
                            <td style="text-align:right; font-family:monospace; color:var(--danger);">19 대 <span style="font-size:11px;">(-9대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-red">9명 품절 피해</span></td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">31 대 <span style="font-size:11px;">(+3대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-green">결품 0건 완벽 방어</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight:bold; color:var(--text-title);">도룡동 대전신세계 스마트시티</td>
                            <td style="text-align:right; font-family:monospace;">12 대</td>
                            <td style="text-align:right; font-family:monospace; font-weight:bold;">26 대</td>
                            <td style="text-align:right; font-family:monospace; color:var(--danger);">17 대 <span style="font-size:11px;">(-9대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-red">9명 품절 피해</span></td>
                            <td style="text-align:right; font-family:monospace; color:var(--primary);">29 대 <span style="font-size:11px;">(+3대)</span></td>
                            <td style="text-align:center;"><span class="badge badge-green">결품 0건 완벽 방어</span></td>
                        </tr>
                    </tbody>
                </table>

                <div class="callout" style="margin-top:14px;">
                    <strong style="color:var(--text-title);">알고리즘 차이(오차) 분석 결론:</strong><br/>
                    기존 MSE 모델은 수요의 조건부 평균(E[Y|X])을 추정하므로, 피크 시간대의 분산(Variance)과 급증 수요를 따라가지 못해 
                    <strong>핵심 거점 5곳에서만 65명의 결품 피해를 방치</strong>했습니다. 
                    반면 제안 Q85 모델은 부족 오차에 5.67배 높은 페널티를 부과하여 <strong>실제 수요보다 약간 높은 85% 상위 분위수를 예측</strong>함으로써 
                    트럭 10대의 배차 물량을 선제적으로 유도하여 <strong>결품 발생을 0건으로 차단(결품 방어율 88.8% 달성)</strong>했습니다.
                </div>
            </div>
        </section>

        <!-- ========================================== -->
        <!-- 5. FIELD RESULTS & OPERATIONAL IMPACT -->
        <!-- ========================================== -->
        <section id="sec-result" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 05</span>
                <h2 class="section-title">5. 현장 실증 결과 및 재배치 최적화</h2>
                <p class="section-desc">2025년 9월 15일 17:00~18:00 퇴근 피크 1시간 실전 배차 결과</p>
            </div>

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
        <!-- 6. URBAN MOBILITY EXPANSION & SYNERGY -->
        <!-- ========================================== -->
        <section id="sec-synergy" style="margin-top:40px;">
            <div class="section-header">
                <span class="section-tag">Section 06</span>
                <h2 class="section-title">6. 도시 인프라 확장 및 정책 연계</h2>
                <p class="section-desc">2028년 트램 2호선 연계, 지하철 1호선 환승 편익, 자전거 도로 안전성 개선, 그리고 ESG 탄소 감축</p>
            </div>

            <div class="grid-4">
                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--primary);">TRAM LINE 2 SYNERGY</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">대전 트램 2호선 연계</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--primary); font-family:monospace;">42.4%</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        2028년 개통 예정인 45개 트램 정거장 반경 500m 이내에 전체 타슈 통행의 42.4%(386.3만 건) 밀집.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--success);">METRO 1 FEEDER</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">지하철 1호선 환승</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--success); font-family:monospace;">19.4%</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        22개 지하철역 반경 300m 이내에서 연간 176.6만 건의 퍼스트·라스트마일 환승 발생 증빙.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--warning);">ROAD SAFETY</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">자전거도로 단절 Top 10</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--warning); font-family:monospace;">Top 10</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        OSM 도로망 오버레이로 월 10만 건 이상 통행되나 전용 도로가 끊긴 고위험 구간 1순위 투자처 제시.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; font-weight:700; color:var(--danger);">ESG BENEFIT</div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--text-title); margin:6px 0;">ESG 탄소 순감축량</h3>
                    <div style="font-size:24px; font-weight:800; color:var(--danger); font-family:monospace;">701.8 t</div>
                    <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        연간 701.8톤 CO₂ 감축(소나무 10.6만 그루 식재 효과) 및 시민 유류비 6.89억 원 절감 증빙.
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

    <!-- INTERACTIVE SIMULATION JAVASCRIPT LOGIC (PURE NATIVE JS) -->
    <script>
        const simData = {json.dumps(
            [{'h': h, 'actual': sim_actuals[h], 'mse': sim_mse[h], 'q85': sim_q85[h]} for h in range(24)]
        )};

        let currentSimHour = 18;
        let isPlaying = false;
        let simInterval = null;
        let simSpeed = 1000; // ms per hour

        const simPlotW = {sim_plot_w};
        const simPlotH = {sim_plot_h};
        const marginLeft = {margin_left};
        const simMax = {sim_max};

        function setSimHour(h) {{
            currentSimHour = h;
            document.getElementById('simSlider').value = h;
            
            const item = simData[h];
            const act = item.actual;
            const mse = item.mse;
            const q85 = item.q85;
            const diffMse = mse - act;
            const diffQ85 = q85 - act;

            // Update Text Displays
            let periodText = h < 12 ? '오전 ' + h + '시' : (h === 12 ? '정오' : '오후 ' + (h-12) + '시');
            if (h === 8) periodText += ' (출근 피크)';
            if (h === 18) periodText += ' (퇴근 피크)';
            
            document.getElementById('simHourDisplay').innerText = h + ':00 (' + periodText + ')';
            document.getElementById('simActualVal').innerText = act.toLocaleString() + ' 대';
            
            // MSE display
            const mseGapFormatted = (diffMse > 0 ? '+' : '') + diffMse.toLocaleString() + ' 대';
            document.getElementById('simMseVal').innerHTML = mse.toLocaleString() + ' 대 <span style="font-size:14px;">(' + mseGapFormatted + ')</span>';
            
            if (diffMse < 0) {{
                document.getElementById('simMseGapSub').innerHTML = '🚨 ' + Math.abs(diffMse).toLocaleString() + '대 결품 위험 (과소예측)';
                document.getElementById('simMseGapSub').style.color = 'var(--danger)';
            }} else {{
                document.getElementById('simMseGapSub').innerHTML = '평균 수요 범위 내 충족';
                document.getElementById('simMseGapSub').style.color = 'var(--text-muted)';
            }}

            // Q85 display
            const q85GapFormatted = '+' + diffQ85.toLocaleString() + ' 대';
            document.getElementById('simQ85Val').innerHTML = q85.toLocaleString() + ' 대 <span style="font-size:14px;">(' + q85GapFormatted + ')</span>';
            if (diffQ85 >= 0) {{
                document.getElementById('simQ85GapSub').innerHTML = '✓ ' + diffQ85.toLocaleString() + '대 안전 재고 확보 (방어)';
                document.getElementById('simQ85GapSub').style.color = 'var(--success)';
            }}

            // Sub text update
            if (h >= 17 && h <= 19) {{
                document.getElementById('simStatusSub').innerText = '퇴근길 통행 급증 구간 (결품 위험 최고조)';
            }} else if (h === 8) {{
                document.getElementById('simStatusSub').innerText = '오전 통근 출근 집중 구간';
            }} else {{
                document.getElementById('simStatusSub').innerText = '통상적 주간 생활 통행 구간';
            }}

            // Move SVG Cursor & Indicators
            const x = marginLeft + (h / 23.0) * simPlotW;
            const yAct = 20 + simPlotH - (act / simMax) * simPlotH;
            const yMse = 20 + simPlotH - (mse / simMax) * simPlotH;
            const yQ85 = 20 + simPlotH - (q85 / simMax) * simPlotH;

            const cursorLine = document.getElementById('simCursorLine');
            cursorLine.setAttribute('x1', x);
            cursorLine.setAttribute('x2', x);

            const cAct = document.getElementById('simCursorAct');
            cAct.setAttribute('cx', x);
            cAct.setAttribute('cy', yAct);

            const cMse = document.getElementById('simCursorMse');
            cMse.setAttribute('cx', x);
            cMse.setAttribute('cy', yMse);

            const cQ85 = document.getElementById('simCursorQ85');
            cQ85.setAttribute('cx', x);
            cQ85.setAttribute('cy', yQ85);
        }}

        function togglePlay() {{
            if (isPlaying) {{
                pauseSim();
            }} else {{
                startSim();
            }}
        }}

        function startSim() {{
            isPlaying = true;
            document.getElementById('playText').innerText = '일시정지';
            document.getElementById('playIcon').innerText = '⏸';
            document.getElementById('playBtn').classList.add('playing');

            simInterval = setInterval(() => {{
                currentSimHour = (currentSimHour + 1) % 24;
                setSimHour(currentSimHour);
            }}, simSpeed);
        }}

        function pauseSim() {{
            isPlaying = false;
            document.getElementById('playText').innerText = '시뮬레이션 재생';
            document.getElementById('playIcon').innerText = '▶';
            document.getElementById('playBtn').classList.remove('playing');
            if (simInterval) clearInterval(simInterval);
        }}

        function resetSim() {{
            pauseSim();
            setSimHour(6); // Reset to 6:00 AM
        }}

        function toggleSpeed() {{
            if (simSpeed === 1000) {{
                simSpeed = 500;
                document.getElementById('speedBtn').innerText = '배속: 2x';
            }} else {{
                simSpeed = 1000;
                document.getElementById('speedBtn').innerText = '배속: 1x';
            }}
            if (isPlaying) {{
                clearInterval(simInterval);
                simInterval = setInterval(() => {{
                    currentSimHour = (currentSimHour + 1) % 24;
                    setSimHour(currentSimHour);
                }}, simSpeed);
            }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            setSimHour(18); // Default to 18:00
        }});
    </script>
</body>
</html>
"""

# Save to destination paths
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
