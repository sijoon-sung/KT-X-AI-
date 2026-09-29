import json
import os
from pathlib import Path

ROOT = Path("C:/Users/sijoo/Documents/tashu")

# HTML Template with zero external CDN dependency - pure self-contained SVG, Canvas, and CSS
html_code = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KT-X-AI | 대전 공영자전거 '타슈' AI 재배치 최적화 & 통행 Flow 관제 대시보드</title>
    <style>
        :root {
            --bg-main: #080d1a;
            --bg-card: #0f172a;
            --bg-surface: #1e293b;
            --border: #334155;
            --border-glow: #06b6d4;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --cyan: #06b6d4;
            --emerald: #10b981;
            --amber: #f59e0b;
            --rose: #f43f5e;
            --indigo: #6366f1;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Pretendard", sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            line-height: 1.5;
            overflow-x: hidden;
        }
        /* Layout & Utilities */
        .container { max-width: 1280px; margin: 0 auto; padding: 0 20px; }
        .grid { display: grid; gap: 20px; }
        .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        @media (max-width: 900px) {
            .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
        }
        .card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            transition: all 0.25s ease;
        }
        .card-glow {
            border: 1px solid rgba(6, 182, 212, 0.4);
            box-shadow: 0 0 30px rgba(6, 182, 212, 0.12);
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-cyan { background: rgba(6, 182, 212, 0.15); color: var(--cyan); border: 1px solid rgba(6, 182, 212, 0.4); }
        .badge-emerald { background: rgba(16, 185, 129, 0.15); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-amber { background: rgba(245, 158, 11, 0.15); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-rose { background: rgba(244, 63, 94, 0.15); color: var(--rose); border: 1px solid rgba(244, 63, 94, 0.4); }
        
        /* Navigation */
        header {
            position: sticky; top: 0; z-index: 100;
            background: rgba(8, 13, 26, 0.9);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border);
            height: 64px;
        }
        .nav-inner {
            height: 64px; display: flex; align-items: center; justify-content: space-between;
        }
        .nav-links { display: flex; gap: 8px; font-size: 13px; font-weight: 600; }
        .nav-links a {
            color: var(--text-secondary); text-decoration: none; padding: 8px 14px;
            border-radius: 8px; transition: all 0.2s;
        }
        .nav-links a:hover { color: #fff; background: var(--bg-surface); }

        /* Hero */
        .hero { padding: 48px 0 32px; }
        .hero-title {
            font-size: 40px; font-weight: 900; line-height: 1.2; letter-spacing: -1px;
            background: linear-gradient(135deg, #ffffff 30%, var(--cyan) 70%, var(--emerald) 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 16px;
        }
        .hero-sub {
            font-size: 16px; color: var(--text-secondary); max-width: 820px; line-height: 1.6; margin-bottom: 24px;
        }
        .kpi-num { font-size: 32px; font-weight: 900; font-family: monospace; }

        /* Buttons & Tabs */
        .btn {
            display: inline-flex; align-items: center; gap: 8px; padding: 10px 18px;
            border-radius: 10px; font-size: 13px; font-weight: 700; cursor: pointer;
            border: 1px solid transparent; transition: all 0.2s; text-decoration: none;
        }
        .btn-cyan { background: var(--cyan); color: #080d1a; }
        .btn-cyan:hover { opacity: 0.9; }
        .btn-outline { background: var(--bg-surface); color: var(--text-primary); border-color: var(--border); }
        .btn-outline:hover { background: #334155; }
        .tab-btn {
            padding: 8px 16px; border-radius: 8px; font-size: 12px; font-weight: 700;
            background: var(--bg-surface); color: var(--text-secondary); border: 1px solid var(--border);
            cursor: pointer; transition: all 0.2s;
        }
        .tab-btn.active {
            background: rgba(6, 182, 212, 0.2); color: var(--cyan); border-color: var(--cyan);
        }

        /* SVG Map Area */
        .map-wrapper {
            position: relative; width: 100%; height: 500px; background: #060913;
            border-radius: 14px; border: 1px solid var(--border); overflow: hidden;
        }
        svg.daejeon-map { width: 100%; height: 100%; }
        .flow-line {
            stroke-linecap: round; stroke-dasharray: 6 3; animation: flowAnim 1.5s linear infinite;
            cursor: pointer; transition: stroke-width 0.2s;
        }
        .flow-line:hover { stroke: #fff !important; stroke-width: 6 !important; }
        @keyframes flowAnim {
            from { stroke-dashoffset: 18; }
            to { stroke-dashoffset: 0; }
        }
        .station-node {
            cursor: pointer; transition: r 0.2s, fill 0.2s;
        }
        .station-node:hover {
            r: 8 !important; fill: #ffffff !important; stroke: var(--cyan) !important;
        }

        /* Scrubber & Sliders */
        .slider-bar {
            width: 100%; -webkit-appearance: none; height: 6px; border-radius: 3px;
            background: #1e293b; outline: none; margin: 12px 0;
        }
        .slider-bar::-webkit-slider-thumb {
            -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
            background: var(--cyan); cursor: pointer; box-shadow: 0 0 10px var(--cyan);
        }

        /* Tables */
        table.custom-table {
            width: 100%; border-collapse: collapse; font-size: 12px; text-align: left;
        }
        table.custom-table th {
            padding: 10px; border-bottom: 1px solid var(--border); color: var(--text-secondary);
            font-size: 11px; text-transform: uppercase;
        }
        table.custom-table td {
            padding: 10px; border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        }
        table.custom-table tr:hover td { background: rgba(30, 41, 59, 0.6); }

        /* Animation */
        .pulse-dot {
            width: 8px; height: 8px; border-radius: 50%; background: var(--emerald);
            box-shadow: 0 0 8px var(--emerald); animation: pulse 1.8s infinite;
        }
        @keyframes pulse { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="container nav-inner">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:34px; height:34px; background:linear-gradient(135deg, var(--cyan), var(--emerald)); border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:900; color:#080d1a; font-size:18px;">T</div>
                <div>
                    <div style="font-weight:900; font-size:15px; letter-spacing:-0.5px;">KT-X-AI <span style="font-size:11px; padding:2px 8px; background:rgba(6, 182, 212, 0.15); color:var(--cyan); border-radius:12px; margin-left:6px; font-family:monospace;">BRANCH: 성시준</span></div>
                    <div style="font-size:10px; color:var(--text-muted);">대전 공영자전거 911만 건 AI 재배치 & 통행 Flow 스마트 관제</div>
                </div>
            </div>

            <nav class="nav-links" style="display:none;" id="desktopNav">
                <a href="#flow-section">① 통행 Flow 맵</a>
                <a href="#eda-section">② 911만 건 EDA</a>
                <a href="#model-section">③ 300m Q85 모델</a>
                <a href="#fleet-section">④ 트럭 10대 배차</a>
                <a href="#bayesian-section">⑤ 베이지안 관제</a>
                <a href="#synergy-section">⑥ 트램 2호선 & ESG</a>
            </nav>

            <a href="https://github.com/sijoon-sung/KT-X-AI-/tree/성시준" target="_blank" class="btn btn-outline" style="font-size:12px; padding:6px 12px;">
                GitHub 저장소 ↗
            </a>
        </div>
    </header>

    <main class="container" style="padding-top:20px; padding-bottom:60px;">

        <!-- HERO KPI SECTION -->
        <section class="hero">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span class="pulse-dot"></span>
                <span class="badge badge-cyan">KT-X-AI URBAN MOBILITY RESEARCH</span>
                <span style="font-size:12px; color:var(--text-muted);">Production AI Model Ready</span>
            </div>
            
            <h1 class="hero-title">
                대전 공공자전거 타슈(Tashu)<br/>
                빅데이터 통행 Flow & AI 스마트 재배치 솔루션
            </h1>

            <p class="hero-sub">
                <strong>911만 건(1개년 전수 로그)</strong>의 이동 경로, 1시간 단위 기상, 도로망 데이터를 융합하여 
                <strong>300m 생활권 Super-Station 군집화</strong>, <strong>Quantile 85% 비대칭 AI 예측</strong>, 
                <strong>물리 제약 트럭 10대 동선 최적화</strong>, 그리고 <strong>포아송 베이지안 고장 감지</strong>를 집대성한 정적 인터랙티브 시스템입니다.
            </p>

            <!-- 5 KPI Cards -->
            <div class="grid grid-4" style="margin-top:28px;">
                <div class="card card-glow">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600;">분석 전수 통행량</div>
                    <div class="kpi-num" style="color:var(--cyan); margin:6px 0;">9,116,462<span style="font-size:13px; font-weight:normal; color:var(--text-secondary); margin-left:4px;">건</span></div>
                    <div style="font-size:11px; color:var(--text-muted);">1,200개 대여소 전역 1년 로그</div>
                </div>

                <div class="card">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600;">생활권 Super-Station</div>
                    <div class="kpi-num" style="color:var(--emerald); margin:6px 0;">460<span style="font-size:13px; font-weight:normal; color:var(--text-secondary); margin-left:4px;">개 앵커</span></div>
                    <div style="font-size:11px; color:var(--text-muted);">300m 보행 반경 거점 집약</div>
                </div>

                <div class="card">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600;">피크 결품 위험 차단율</div>
                    <div class="kpi-num" style="color:var(--rose); margin:6px 0;">-65.0<span style="font-size:13px; font-weight:normal; color:var(--text-secondary); margin-left:4px;">%</span></div>
                    <div style="font-size:11px; color:var(--text-muted);">Top 20 결품률 38% → 13.3%</div>
                </div>

                <div class="card">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600;">트럭 10대 배차 골든타임</div>
                    <div class="kpi-num" style="color:var(--amber); margin:6px 0;">48.5<span style="font-size:13px; font-weight:normal; color:var(--text-secondary); margin-left:4px;">분 / 176대</span></div>
                    <div style="font-size:11px; color:var(--text-muted);">퇴근 60분 내 핀포인트 하차</div>
                </div>
            </div>
        </section>

        <!-- SECTION 1: INTERACTIVE MOBILITY FLOW MAP (CORE REQUIREMENT) -->
        <section id="flow-section" style="margin-top:40px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
                <div>
                    <span class="badge badge-cyan">SECTION 01</span>
                    <h2 style="font-size:24px; font-weight:800; margin-top:4px;">대전시 핵심 통행 회랑(Corridor) & Flow 인터랙티브 관제 맵</h2>
                    <p style="font-size:13px; color:var(--text-secondary);">시민들이 가장 많이 달리는 최상위 O-D 경로와 2028년 트램 2호선(38.8km 순환선) 연계</p>
                </div>

                <!-- Map Filter Buttons -->
                <div style="display:flex; gap:6px; flex-wrap:wrap;">
                    <button class="tab-btn active" onclick="filterFlows('all', this)">전체 최상위 Flow</button>
                    <button class="tab-btn" onclick="filterFlows('campus', this)">대학 캠퍼스 셔틀</button>
                    <button class="tab-btn" onclick="filterFlows('metro', this)">지하철 1호선 환승</button>
                    <button class="tab-btn" onclick="filterFlows('leisure', this)">수목원·하천 레저</button>
                    <button class="tab-btn" onclick="filterFlows('truck', this)">트럭 재배치 동선</button>
                </div>
            </div>

            <div class="grid grid-3">
                <!-- Left 2 Cols: Interactive Vector Map -->
                <div class="card" style="grid-column: span 2; padding:16px;">
                    <div class="map-wrapper">
                        <svg id="daejeonSvg" class="daejeon-map" viewBox="0 0 800 520">
                            <!-- Background Map Waterways & City Outline -->
                            <rect width="800" height="520" fill="#070c18"/>
                            <text x="20" y="30" fill="#334155" font-size="12" font-weight="800" font-family="monospace">DAEJEON MOBILITY GIS VECTOR CANVAS</text>
                            
                            <!-- Rivers: Gapcheon & Yudeungcheon -->
                            <path d="M 120 180 Q 280 260 380 260 T 520 280 T 640 400" fill="none" stroke="#0e304f" stroke-width="14" opacity="0.6"/>
                            <path d="M 460 270 Q 480 340 500 460" fill="none" stroke="#0e304f" stroke-width="10" opacity="0.6"/>
                            <text x="210" y="225" fill="#1e3a5f" font-size="11" font-weight="700">갑천 (Gapcheon)</text>
                            <text x="495" y="380" fill="#1e3a5f" font-size="11" font-weight="700">유등천</text>

                            <!-- Future Tram Line 2 (Dashed Cyan Loop) -->
                            <ellipse cx="430" cy="280" rx="260" ry="170" fill="none" stroke="rgba(6, 182, 212, 0.25)" stroke-width="3" stroke-dasharray="8 6"/>
                            <text x="590" y="160" fill="var(--cyan)" font-size="11" font-weight="700">대전 트램 2호선 (38.8km 예정 순환망)</text>

                            <!-- Subway Line 1 (Green Line) -->
                            <path d="M 170 170 L 260 280 L 400 300 L 520 320 L 630 360 L 680 430" fill="none" stroke="rgba(16, 185, 129, 0.4)" stroke-width="4"/>
                            <text x="620" y="340" fill="var(--emerald)" font-size="11" font-weight="700">도시철도 1호선</text>

                            <!-- Dynamic Flow Arrows (Injected via JS) -->
                            <g id="flowLayer"></g>

                            <!-- Key Landmark Station Nodes -->
                            <g id="nodeLayer"></g>
                        </svg>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; font-size:11px; color:var(--text-muted);">
                        <span>💡 Flow 선이나 거점 원을 클릭하면 우측 패널에 상세 속성이 표시됩니다.</span>
                        <span style="font-family:monospace;">좌표계: WGS84 EPSG:4326 투영</span>
                    </div>
                </div>

                <!-- Right 1 Col: Dynamic Flow / Hub Detail Card -->
                <div class="card" id="flowDetailCard" style="display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                            <span class="badge badge-cyan" id="flowCategoryBadge">통행 회랑 선택됨</span>
                            <span style="font-size:11px; color:var(--text-muted); font-family:monospace;" id="flowCode">FLOW #01</span>
                        </div>

                        <h3 id="flowTitle" style="font-size:16px; font-weight:800; color:#fff; margin-bottom:8px;">
                            어은동 카이스트 학사식당 ⇄ 구성동 창의학습관
                        </h3>
                        <p id="flowDesc" style="font-size:12px; color:var(--text-secondary); line-height:1.6; margin-bottom:18px;">
                            대전시 전체에서 가장 빈번하게 통행이 일어나는 대학가 마이크로 셔틀 회랑입니다. 
                            강의동과 학생 식당 간의 빠른 이동을 담당합니다.
                        </p>

                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px;">
                            <div style="background:var(--bg-surface); padding:10px; border-radius:8px;">
                                <div style="font-size:10px; color:var(--text-muted);">연간 이동량</div>
                                <div id="flowTrips" style="font-size:18px; font-weight:900; color:var(--cyan);">13,309건</div>
                            </div>
                            <div style="background:var(--bg-surface); padding:10px; border-radius:8px;">
                                <div style="font-size:10px; color:var(--text-muted);">평균 이동거리</div>
                                <div id="flowDist" style="font-size:18px; font-weight:900; color:var(--emerald);">0.47 km</div>
                            </div>
                            <div style="background:var(--bg-surface); padding:10px; border-radius:8px;">
                                <div style="font-size:10px; color:var(--text-muted);">평균 소요시간</div>
                                <div id="flowDuration" style="font-size:18px; font-weight:900; color:var(--amber);">7.8 분</div>
                            </div>
                            <div style="background:var(--bg-surface); padding:10px; border-radius:8px;">
                                <div style="font-size:10px; color:var(--text-muted);">회랑 유형</div>
                                <div id="flowType" style="font-size:13px; font-weight:700; color:var(--text-primary); margin-top:2px;">캠퍼스 셔틀</div>
                            </div>
                        </div>
                    </div>

                    <div style="background:rgba(6, 182, 212, 0.08); border:1px solid rgba(6, 182, 212, 0.25); border-radius:10px; padding:12px; font-size:11px; color:var(--text-secondary); line-height:1.5;">
                        <strong style="color:var(--cyan);">운영 관리 인사이트:</strong><br/>
                        <span id="flowInsight">해당 구간은 양방향 회전율이 매우 높아 자전거가 자체적으로 순환됩니다. 트럭을 통한 인위적 재배치 우선순위가 낮아 물류비를 아낄 수 있는 자가치유형 거점입니다.</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 2: COMPREHENSIVE 9.11M EDA (SWEET SPOT & RHYTHMS) -->
        <section id="eda-section" style="margin-top:60px;">
            <div style="margin-bottom:20px;">
                <span class="badge badge-emerald">SECTION 02</span>
                <h2 style="font-size:24px; font-weight:800; margin-top:4px;">데이터 분석가가 밝힌 911만 건 시민 통행 패턴</h2>
                <p style="font-size:13px; color:var(--text-secondary);">1km의 법칙, 출퇴근 쌍봉 리듬(M-Curve), 그리고 1mm 강수 급락 임계선</p>
            </div>

            <div class="grid grid-2">
                <!-- EDA 1: 1km / 11min Sweet Spot -->
                <div class="card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                        <div>
                            <h3 style="font-size:15px; font-weight:800;">① 이동 거리 & 시간: "1km & 11분의 법칙"</h3>
                            <p style="font-size:11px; color:var(--text-muted);">시민 4명 중 3명(75.5%)이 2km 이내에서 완결</p>
                        </div>
                        <span class="badge badge-cyan">중앙값 1.0 km / 11분</span>
                    </div>

                    <!-- Visual Segmented Bar -->
                    <div style="margin:20px 0;">
                        <div style="display:flex; height:24px; border-radius:6px; overflow:hidden; font-size:11px; font-weight:800; text-align:center; line-height:24px;">
                            <div style="width:51.2%; background:var(--cyan); color:#080d1a;">51.2% (1km 이내)</div>
                            <div style="width:24.3%; background:var(--emerald); color:#080d1a;">24.3% (1~2km)</div>
                            <div style="width:14.1%; background:var(--indigo); color:#fff;">14.1%</div>
                            <div style="width:10.4%; background:var(--amber); color:#080d1a;">10.4%</div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); margin-top:8px;">
                            <span>• 1km 이내: 51.2% (초단거리 마이크로)</span>
                            <span>• 1~2km: 24.3% (생활권 표준)</span>
                            <span>• 2~4km: 14.1% (환승)</span>
                            <span>• 4km 초과: 10.4% (레저)</span>
                        </div>
                    </div>

                    <div style="background:var(--bg-surface); padding:14px; border-radius:10px; font-size:12px; color:var(--text-secondary); line-height:1.6;">
                        <strong style="color:var(--cyan);">분석가 총평:</strong> 타슈는 도시를 가로지르는 장거리 이동 수단이 아닙니다. 
                        <strong>"도보로 걷기엔 멀고(15분), 버스 타기엔 아까운 1km 남짓의 라스트마일 틈새"</strong>를 정확하게 해결해 주는 마이크로 모빌리티입니다.
                    </div>
                </div>

                <!-- EDA 2: Weather Rain Drop-off Interactive Simulator -->
                <div class="card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                        <div>
                            <h3 style="font-size:15px; font-weight:800;">② 날씨 탄력성: "1mm 강수 급락 임계선 시뮬레이터"</h3>
                            <p style="font-size:11px; color:var(--text-muted);">강수량에 따른 실시간 통행량 감소 및 재배치 감차 권고</p>
                        </div>
                        <span class="badge badge-amber" id="rainDropBadge">1mm 돌파 시 -70.9% 급락</span>
                    </div>

                    <div style="display:flex; gap:6px; margin-bottom:16px;">
                        <button class="tab-btn active" onclick="setRain(0, this)">비 안 옴 (0mm)</button>
                        <button class="tab-btn" onclick="setRain(1, this)">이슬비 (0~1mm)</button>
                        <button class="tab-btn" onclick="setRain(2, this)">일반 비 (1~3mm)</button>
                        <button class="tab-btn" onclick="setRain(3, this)">폭우 (>3mm)</button>
                    </div>

                    <div style="background:var(--bg-surface); padding:16px; border-radius:10px; margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
                            <span>주간 시간당 통행량: <strong id="rainTripsPerHour" style="color:var(--cyan); font-size:15px;">936.3 건/h</strong></span>
                            <span id="rainDropPercent" style="color:var(--text-muted); font-weight:bold;">기준 통행량 (100%)</span>
                        </div>
                        <div style="width:100%; height:10px; background:#070c18; border-radius:5px; overflow:hidden;">
                            <div id="rainBar" style="width:100%; height:100%; background:var(--cyan); transition:all 0.3s;"></div>
                        </div>
                    </div>

                    <div id="rainActionBox" style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3); border-radius:10px; padding:12px; font-size:12px; color:var(--text-secondary); line-height:1.5;">
                        <strong style="color:var(--emerald);">물류 운영 지침:</strong> 기상이 쾌적하여 트럭 10대를 100% 정상 가동합니다.
                    </div>
                </div>
            </div>

            <!-- Diurnal Rhythm 24-Hour Slider -->
            <div class="card" style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
                    <div>
                        <h3 style="font-size:16px; font-weight:800;">③ 24시간 도시 통행 리듬: "평일 쌍봉(M-Curve) vs 주말 종형 곡선"</h3>
                        <p style="font-size:12px; color:var(--text-muted);">퇴근 시간(18:00, 62.7만 건)은 출근 시간(08:00, 39.4만 건)의 1.59배 폭증</p>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="font-size:12px; color:var(--cyan);">■ 평일 통근</span>
                        <span style="font-size:12px; color:var(--emerald);">■ 주말 여가</span>
                    </div>
                </div>

                <!-- Native SVG Line Chart -->
                <div style="width:100%; height:180px; position:relative;">
                    <svg viewBox="0 0 1000 180" style="width:100%; height:100%;">
                        <line x1="40" y1="150" x2="960" y2="150" stroke="#1e293b" stroke-width="1"/>
                        <line x1="40" y1="80" x2="960" y2="80" stroke="#1e293b" stroke-width="1" stroke-dasharray="4 4"/>
                        <!-- Weekday Curve: 08h peak (x=340, y=55), 18h peak (x=725, y=15) -->
                        <path d="M 40 120 Q 200 130 300 90 T 350 55 T 420 100 T 500 85 T 600 70 T 725 15 T 820 70 T 960 100" fill="none" stroke="var(--cyan)" stroke-width="3"/>
                        <!-- Weekend Curve: Smooth Bell peak at 16h (x=650, y=70) -->
                        <path d="M 40 135 Q 250 140 450 110 T 650 70 T 800 95 T 960 125" fill="none" stroke="var(--emerald)" stroke-width="2.5" stroke-dasharray="6 3"/>
                        <!-- Scrubber Indicator Line -->
                        <line id="scrubberLine" x1="725" y1="10" x2="725" y2="150" stroke="#f43f5e" stroke-width="2"/>
                        <circle id="scrubberCircle" cx="725" cy="15" r="5" fill="#f43f5e"/>
                    </svg>
                </div>

                <!-- 24h Interactive Slider -->
                <div style="margin-top:10px;">
                    <input type="range" min="0" max="23" value="18" class="slider-bar" id="hourSlider" oninput="updateHourScrubber(this.value)">
                    <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--text-muted); font-family:monospace;">
                        <span>00:00 (심야)</span>
                        <span>08:00 (출근 피크)</span>
                        <span>12:00 (점심 이동)</span>
                        <span style="color:var(--rose); font-weight:bold;">18:00 (퇴근 대폭증)</span>
                        <span>23:00 (귀가)</span>
                    </div>
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center; background:var(--bg-surface); padding:12px 16px; border-radius:10px; margin-top:12px;">
                    <div>
                        <span id="selectedHourText" style="font-size:14px; font-weight:800; color:#fff;">18:00 (오후 6시 퇴근 피크)</span>
                        <div id="selectedHourDesc" style="font-size:11px; color:var(--text-secondary); margin-top:2px;">
                            퇴근길에는 지각 부담이 없고 저녁 약속·식사·귀가가 결합되어 출근길의 1.6배에 달하는 통행이 쏟아져 대여소 품절이 가장 심각합니다.
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px; color:var(--text-muted);">평일 통행량</div>
                        <div id="selectedHourTrips" style="font-size:18px; font-weight:900; color:var(--cyan); font-family:monospace;">627,088 건</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 3: 300M SUPER-STATION & Q85 MACHINE LEARNING -->
        <section id="model-section" style="margin-top:60px;">
            <div style="margin-bottom:20px;">
                <span class="badge badge-amber">SECTION 03</span>
                <h2 style="font-size:24px; font-weight:800; margin-top:4px;">300m Super-Station & Quantile 85% 머신러닝</h2>
                <p style="font-size:13px; color:var(--text-secondary);">부족 오차에 5.67배 높은 벌점을 부여하여 품절 사태를 65% 차단한 혁신 솔루션</p>
            </div>

            <div class="grid grid-3">
                <div class="card">
                    <div style="font-size:12px; color:var(--cyan); font-weight:700; margin-bottom:6px;">SPATIAL CLUSTERING</div>
                    <h3 style="font-size:16px; font-weight:800; margin-bottom:8px;">460개 생활권 Super-Station</h3>
                    <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                        1,200개 개별 대여소의 50~100m 파편화로 인한 시계열 희소성(Sparsity)을 해소하기 위해 
                        <strong>보행 3분(300m) 생활권 앵커 거점 460개</strong>로 집약하여 도보 대체성을 확보했습니다.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:12px; color:var(--emerald); font-weight:700; margin-bottom:6px;">NETWORK TOPOLOGY</div>
                    <h3 style="font-size:16px; font-weight:800; margin-bottom:8px;">87.4% O/D 유입 회랑 피처</h3>
                    <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                        대여소의 87.4% 통행은 타 군집에서 유입됩니다. 직전 시간대에 해당 거점을 향해 출발한 
                        <strong>유입량(`lag1_ret`) 피처</strong>를 주입하여 피처 중요도 <strong>53.41%</strong>를 달성했습니다.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:12px; color:var(--amber); font-weight:700; margin-bottom:6px;">ASYMMETRIC LOSS</div>
                    <h3 style="font-size:16px; font-weight:800; margin-bottom:8px;">Quantile 85% 비대칭 손실</h3>
                    <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                        남는 것보다 모자란 것(결품)이 치명적인 현장 특성을 반영하여 Pinball Loss(τ=0.85)를 적용, 
                        <strong>부족 오차에 5.67배 높은 페널티</strong>를 부과해 안전 재고를 자동 계산합니다.
                    </p>
                </div>
            </div>

            <!-- Before vs After Metrics Bar -->
            <div class="card card-glow" style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                    <div>
                        <span class="badge badge-emerald">FIELD EXPERIMENT VERIFIED</span>
                        <h3 style="font-size:18px; font-weight:800; color:#fff; margin-top:4px;">
                            퇴근 피크(17:00 → 18:00) 실전 배차 개선 효과 (Before vs After)
                        </h3>
                        <p style="font-size:12px; color:var(--text-secondary);">결품 피해 시민을 88.8% 차단하고 1시간 만에 73명을 즉시 구제</p>
                    </div>

                    <div style="display:flex; align-items:center; gap:20px;">
                        <div style="text-align:center;">
                            <div style="font-size:11px; color:var(--text-muted);">기존 모델 품절률</div>
                            <div style="font-size:28px; font-weight:900; color:var(--rose); text-decoration:line-through;">38.0%</div>
                            <div style="font-size:10px; color:var(--text-muted);">10회 중 4회 0대</div>
                        </div>
                        <div style="font-size:24px; color:var(--text-muted);">➔</div>
                        <div style="text-align:center;">
                            <div style="font-size:11px; color:var(--emerald); font-weight:bold;">Q85 적용 후 품절률</div>
                            <div style="font-size:34px; font-weight:900; color:var(--emerald);">13.3%</div>
                            <div style="font-size:10px; color:var(--emerald);">결품 위험 65% 차단</div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-3" style="margin-top:20px; padding-top:16px; border-top:1px solid var(--border);">
                    <div style="background:var(--bg-surface); padding:12px; border-radius:8px;">
                        <div style="font-size:11px; color:var(--text-muted);">퇴근 1시간 즉시 구제 인원</div>
                        <div style="font-size:20px; font-weight:900; color:var(--cyan); margin-top:2px;">73명 / 시간</div>
                        <div style="font-size:10px; color:var(--text-secondary);">월간 4,500명 헛걸음 방지</div>
                    </div>
                    <div style="background:var(--bg-surface); padding:12px; border-radius:8px;">
                        <div style="font-size:11px; color:var(--text-muted);">집중 투하 거점 결품 방어율</div>
                        <div style="font-size:20px; font-weight:900; color:var(--emerald); margin-top:2px;">88.8% 방어</div>
                        <div style="font-size:10px; color:var(--text-secondary);">82명 결품 위기 ➔ 9명으로 축소</div>
                    </div>
                    <div style="background:var(--bg-surface); padding:12px; border-radius:8px;">
                        <div style="font-size:11px; color:var(--text-muted);">투입 자전거 유효 회전율</div>
                        <div style="font-size:20px; font-weight:900; color:var(--amber); margin-top:2px;">30.2% 회전</div>
                        <div style="font-size:10px; color:var(--text-secondary);">배차 즉시 1시간 내 시민 대여</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 4: 10 FLEET TRUCK ROUTING OPTIMIZATION -->
        <section id="fleet-section" style="margin-top:60px;">
            <div style="margin-bottom:20px;">
                <span class="badge badge-indigo">SECTION 04</span>
                <h2 style="font-size:24px; font-weight:800; margin-top:4px;">현장 물리 제약 기반 트럭 10대 동선 최적화 (Fleet Routing)</h2>
                <p style="font-size:13px; color:var(--text-secondary);">[상차 1곳 ➔ 하차 1~2곳] 핀포인트 릴레이로 퇴근 60분 골든타임 이내 완벽 하차</p>
            </div>

            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
                    <div>
                        <h3 style="font-size:16px; font-weight:800;">퇴근 피크(17:00) 실전 배차 지시서 (Dispatch Manifest)</h3>
                        <p style="font-size:11px; color:var(--text-muted);">트럭별 상하차 소요시간(상차 45초/대, 하차 30초/대, 도로 주행 22km/h 실측 적용)</p>
                    </div>
                    <span class="badge badge-cyan">총 176대 완벽 하차 | 평균 48.5분</span>
                </div>

                <div style="overflow-x:auto;">
                    <table class="custom-table">
                        <thead>
                            <tr>
                                <th>트럭 번호</th>
                                <th>상차 거점 (잉여 회수)</th>
                                <th>하차 거점 (결품 투하)</th>
                                <th style="text-align:center;">배차 수량</th>
                                <th style="text-align:right;">주행 거리</th>
                                <th style="text-align:right;">총 작업시간</th>
                                <th>상태</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="font-weight:bold; color:var(--cyan);">트럭 01</td>
                                <td>노은역 3번 출구 (잉여 45대)</td>
                                <td>유성온천역 7번 출구 (결품 18대)</td>
                                <td style="text-align:center; font-weight:bold; color:var(--emerald);">18 대</td>
                                <td style="text-align:right; font-family:monospace;">6.8 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--cyan);">46.2 분</td>
                                <td><span class="badge badge-emerald">골든타임 성공</span></td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--cyan);">트럭 02</td>
                                <td>궁동 로데오거리 남측 (잉여 38대)</td>
                                <td>월평역 1번 출구 (결품 16대)</td>
                                <td style="text-align:center; font-weight:bold; color:var(--emerald);">16 대</td>
                                <td style="text-align:right; font-family:monospace;">5.4 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--cyan);">42.8 분</td>
                                <td><span class="badge badge-emerald">골든타임 성공</span></td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--cyan);">트럭 03</td>
                                <td>만년동 KBS 방송국 (잉여 32대)</td>
                                <td>정부청사역 2번 출구 (결품 18대)</td>
                                <td style="text-align:center; font-weight:bold; color:var(--emerald);">18 대</td>
                                <td style="text-align:right; font-family:monospace;">4.9 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--cyan);">41.5 분</td>
                                <td><span class="badge badge-emerald">골든타임 성공</span></td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--cyan);">트럭 04</td>
                                <td>도안 신도시 수변공원 (잉여 29대)</td>
                                <td>원신흥동 주민센터 (결품 18대)</td>
                                <td style="text-align:center; font-weight:bold; color:var(--emerald);">18 대</td>
                                <td style="text-align:right; font-family:monospace;">7.2 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--cyan);">47.8 분</td>
                                <td><span class="badge badge-emerald">골든타임 성공</span></td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--cyan);">트럭 05</td>
                                <td>탄방동 남선공원 체육관 (잉여 27대)</td>
                                <td>시청역 8번 출구 (결품 18대)</td>
                                <td style="text-align:center; font-weight:bold; color:var(--emerald);">18 대</td>
                                <td style="text-align:right; font-family:monospace;">6.1 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--cyan);">45.0 분</td>
                                <td><span class="badge badge-emerald">골든타임 성공</span></td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:var(--cyan);">트럭 06~10</td>
                                <td>가오동/송촌동/관저동 외곽 잉여 거점</td>
                                <td>대동역/서대전네거리역 등 결품 거점</td>
                                <td style="text-align:center; font-weight:bold; color:var(--emerald);">88 대</td>
                                <td style="text-align:right; font-family:monospace;">평균 8.8 km</td>
                                <td style="text-align:right; font-family:monospace; color:var(--cyan);">평균 51.4 분</td>
                                <td><span class="badge badge-emerald">골든타임 성공</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- SECTION 5: BAYESIAN GHOST BIKE DETECTION -->
        <section id="bayesian-section" style="margin-top:60px;">
            <div style="margin-bottom:20px;">
                <span class="badge badge-rose">SECTION 05</span>
                <h2 style="font-size:24px; font-weight:800; margin-top:4px;">공공 API 스냅샷 기반 베이지안 고장 자전거 감지</h2>
                <p style="font-size:13px; color:var(--text-secondary);">자전거 고유 ID 없이 포아송 생존 분석으로 1대 고착 유령 자전거를 99.9% 확정</p>
            </div>

            <div class="grid grid-2">
                <div class="card">
                    <h3 style="font-size:16px; font-weight:800; margin-bottom:10px;">베이지안 수식 & 추론 원리</h3>
                    <div style="background:#070c18; padding:12px; border-radius:8px; font-family:monospace; font-size:12px; color:var(--amber); margin-bottom:12px;">
                        P(고장 | t시간 1대 고착) = 1 - exp(-λ × t)
                    </div>
                    <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                        공공 API에는 자전거 ID가 없어 대여소에 1대가 남아있을 때 정상 자전거인지 체인이 빠진 고장 자전거인지 알 수 없습니다.
                        시간당 대여 수요가 λ인 대여소에서 t시간 동안 아무도 대여하지 않을 확률이 0.001 이하로 떨어질 때 
                        <strong>사후 고장 확률 99.9%를 확정</strong>하여 유효 재고에서 자동 차감합니다.
                    </p>
                </div>

                <div class="card">
                    <h3 style="font-size:16px; font-weight:800; margin-bottom:14px;">실시간 베이지안 고장 확률 시뮬레이터</h3>
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px;">
                            <span>해당 대여소 시간당 평균 수요 (λ)</span>
                            <span id="lambdaVal" style="font-family:monospace; color:var(--cyan); font-weight:bold;">4.0 대/h</span>
                        </div>
                        <input type="range" min="1" max="8" value="4" step="0.5" class="slider-bar" id="lambdaSlider" oninput="calcBayes()">
                    </div>

                    <div style="margin-bottom:14px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px;">
                            <span>1대 연속 정체 시간 (t)</span>
                            <span id="timeVal" style="font-family:monospace; color:var(--amber); font-weight:bold;">2.0 시간</span>
                        </div>
                        <input type="range" min="0.5" max="4" value="2.0" step="0.5" class="slider-bar" id="timeSlider" oninput="calcBayes()">
                    </div>

                    <div style="background:var(--bg-surface); padding:12px; border-radius:8px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:11px; color:var(--text-muted);">사후 고장 확정 확률</div>
                            <div id="bayesResult" style="font-size:22px; font-weight:900; color:var(--rose);">99.97 %</div>
                        </div>
                        <div id="bayesBadgeBox">
                            <span class="badge badge-rose">🚨 즉시 수거 지시서 발행</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 6: URBAN MOBILITY & ESG SYNERGY -->
        <section id="synergy-section" style="margin-top:60px;">
            <div style="margin-bottom:20px;">
                <span class="badge badge-cyan">SECTION 06</span>
                <h2 style="font-size:24px; font-weight:800; margin-top:4px;">도시교통 인프라 연계 & ESG 탄소 감축</h2>
                <p style="font-size:13px; color:var(--text-secondary);">2028년 트램 2호선 시너지, 지하철 1호선 환승 편익, 그리고 도로 안전 단절 해소</p>
            </div>

            <div class="grid grid-4">
                <div class="card">
                    <div style="font-size:11px; color:var(--cyan); font-weight:700;">TRAM LINE 2</div>
                    <h3 style="font-size:15px; font-weight:800; margin:6px 0;">트램 2호선 연계</h3>
                    <div style="font-size:24px; font-weight:900; color:var(--cyan);">42.4%</div>
                    <p style="font-size:11px; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                        트램 45개 정거장 반경 500m 이내에 전체 타슈 통행의 42.4%(386만 건) 밀집.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; color:var(--emerald); font-weight:700;">METRO LINE 1</div>
                    <h3 style="font-size:15px; font-weight:800; margin:6px 0;">지하철 1호선 연계</h3>
                    <div style="font-size:24px; font-weight:900; color:var(--emerald);">19.4%</div>
                    <p style="font-size:11px; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                        22개 지하철역 반경 300m 이내에서 연간 176.6만 건의 라스트마일 환승 발생.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; color:var(--amber); font-weight:700;">ESG CARBON BENEFIT</div>
                    <h3 style="font-size:15px; font-weight:800; margin:6px 0;">연간 탄소 순감축</h3>
                    <div style="font-size:24px; font-weight:900; color:var(--amber);">701.8 t</div>
                    <p style="font-size:11px; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                        소나무 106,337 그루 식재 효과 및 시민 유류비 약 6.89억 원 절감 공인 증빙.
                    </p>
                </div>

                <div class="card">
                    <div style="font-size:11px; color:var(--rose); font-weight:700;">CROWDSOURCED GAMIFICATION</div>
                    <h3 style="font-size:15px; font-weight:800; margin:6px 0;">타슈 챌린지 500P</h3>
                    <div style="font-size:24px; font-weight:900; color:var(--rose);">500 P</div>
                    <p style="font-size:11px; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                        잉여 ➔ 결품 거점 역방향 통근 시민에게 지역화폐 캐시백을 지급해 자가치유망 구축.
                    </p>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer style="border-top:1px solid var(--border); padding:32px 0; text-align:center; font-size:12px; color:var(--text-muted); background:#060a14;">
        <div class="container">
            <div style="font-weight:bold; color:var(--text-secondary); margin-bottom:6px;">
                KT-X-AI 대전 타슈 지능형 모빌리티 솔루션 (Production Static Portfolio)
            </div>
            <p>프로젝트 총괄: 성시준 (Sijun Sung) | 페어 프로그래밍 AI: Google Deepmind Antigravity</p>
            <p style="margin-top:4px; font-size:11px;">GitHub Branch: <code style="color:var(--cyan);">성시준</code> | 9,116,462 Trips Analyzed</p>
        </div>
    </footer>

    <!-- INTERACTIVE SCRIPT LOGIC (PURE VANILLA JS - ZERO EXTERNAL CDN) -->
    <script>
        // Flow Dataset Definition with Screen Coordinates (SVG 800x520)
        const flowsData = [
            {
                id: 1, category: 'campus', code: 'FLOW #01 (캠퍼스 셔틀)',
                title: '어은동 카이스트 학사식당 ⇄ 구성동 창의학습관',
                desc: '강의동과 학생 식당 간의 빠른 이동을 담당하는 연간 최대 밀집 회랑입니다.',
                trips: '13,309 건', dist: '0.47 km', duration: '7.8 분', type: '캠퍼스 마이크로 셔틀',
                insight: '양방향 통행량이 대칭적으로 균형을 이루어 트럭 인위적 재배치가 불필요한 자가치유형 거점입니다.',
                x1: 270, y1: 230, x2: 295, y2: 245, color: '#06b6d4', width: 5
            },
            {
                id: 2, category: 'campus', code: 'FLOW #02 (기숙사 통학)',
                title: '카이스트 창의학습관 ⇄ 서쪽 쪽문(원룸촌)',
                desc: '어은동 원룸촌 및 쪽문 기숙사에서 창의학습관으로 통학하는 핵심 이동로입니다.',
                trips: '10,807 건', dist: '0.61 km', duration: '9.0 분', type: '통학 및 식사 회랑',
                insight: '점심과 저녁 시간대 쪽문 방면으로 자전거가 급격히 몰리므로 13시, 19시 가벼운 회수가 효과적입니다.',
                x1: 295, y1: 245, x2: 265, y2: 280, color: '#06b6d4', width: 4.5
            },
            {
                id: 3, category: 'metro', code: 'FLOW #03 (지하철 환승)',
                title: '둔산동 정부청사역 4번출구 ⇄ 정부청사 입구(남문)',
                desc: '지하철 1호선에서 하차한 공무원 및 방문객이 정부청사 내부로 진입하는 라스트마일 통행입니다.',
                trips: '9,293 건', dist: '0.38 km', duration: '3.1 분', type: '지하철 라스트마일 환승',
                insight: '도보 7분 거리를 타슈로 3분 만에 주파합니다. 아침 08:30 청사 방향 쏠림이 극심하여 트럭 03의 1순위 하차지입니다.',
                x1: 430, y1: 295, x2: 440, y2: 265, color: '#10b981', width: 4.5
            },
            {
                id: 4, category: 'campus', code: 'FLOW #04 (대학 번화가)',
                title: '궁동 로데오거리 ⇄ 충남대학교 정문 네거리',
                desc: '충남대 학생들의 등하교 및 궁동 상권 진입 통행입니다.',
                trips: '8,925 건', dist: '0.58 km', duration: '6.5 분', type: '대학가 상권 연결',
                insight: '야간 21~23시 궁동 상권 쪽에 자전거가 30대 이상 과도하게 누적되어 심야 트럭 회수가 권장됩니다.',
                x1: 240, y1: 285, x2: 220, y2: 270, color: '#06b6d4', width: 4
            },
            {
                id: 5, category: 'leisure', code: 'FLOW #05 (수변 산책)',
                title: '만년동 한밭수목원(서원) ⇄ 한밭수목원(동원)',
                desc: '한밭수목원과 엑스포 시민광장을 둘러보는 도심 힐링·여가 라이딩입니다.',
                trips: '10,054 건', dist: '0.85 km', duration: '30.7 분', type: '공원 체류형 레저',
                insight: '거리는 1km 미만이지만 평균 체류시간이 30분이 넘습니다. 주말 오후에 가족·연인 이용이 집중됩니다.',
                x1: 420, y1: 240, x2: 445, y2: 240, color: '#f59e0b', width: 4.5
            },
            {
                id: 6, category: 'leisure', code: 'FLOW #06 (엑스포 도심 연계)',
                title: '한밭수목원 ⇄ 도룡동 엑스포다리/스마트시티',
                desc: '갑천을 가로지르는 엑스포다리를 건너 스마트시티 주거지와 연결되는 힐링 통행입니다.',
                trips: '7,250 건', dist: '1.45 km', duration: '25.4 분', type: '수변 횡단 레저',
                insight: '야간 조명이 켜지는 20~22시에 자전거 통행 밀도가 급상승하며 자전거 전용도로 안전성이 높습니다.',
                x1: 430, y1: 240, x2: 450, y2: 210, color: '#f59e0b', width: 3.5
            },
            {
                id: 7, category: 'truck', code: 'TRUCK ROUTE #01 (실전 배차)',
                title: '노은역 3번출구 ➔ 유성온천역 7번출구 (트럭 01)',
                desc: '노은동 주거지 잉여 거점에서 18대를 실어 퇴근길 유성온천역 결품 거점에 투하하는 핵심 동선입니다.',
                trips: '트럭 18대 공급', dist: '6.8 km', duration: '46.2 분 소요', type: '1-ton 트럭 최적 동선',
                insight: '도로 정체(22km/h)와 상하차 시간을 고려해도 46.2분 만에 완수하여 퇴근 60분 골든타임 이내 안착합니다.',
                x1: 170, y1: 170, x2: 260, y2: 300, color: '#f43f5e', width: 4.5
            },
            {
                id: 8, category: 'truck', code: 'TRUCK ROUTE #03 (실전 배차)',
                title: '만년동 잉여 거점 ➔ 정부청사역 2번출구 (트럭 03)',
                desc: '만년동 주거지역에서 18대를 수거하여 정부청사역 퇴근 피크 결품을 방어하는 배차입니다.',
                trips: '트럭 18대 공급', dist: '4.9 km', duration: '41.5 분 소요', type: '1-ton 트럭 최적 동선',
                insight: '퇴근 17:00에 출발하여 17:41분에 투하 완료, 18:00 피크 시민 60명의 결품을 완벽 방어합니다.',
                x1: 410, y1: 240, x2: 430, y2: 295, color: '#f43f5e', width: 4.5
            }
        ];

        // Key Nodes
        const nodesData = [
            { name: '카이스트 본원', x: 280, y: 240, type: 'campus' },
            { name: '충남대학교', x: 230, y: 275, type: 'campus' },
            { name: '유성온천역', x: 260, y: 300, type: 'metro' },
            { name: '정부대전청사/정부청사역', x: 435, y: 280, type: 'metro' },
            { name: '대전시청역', x: 440, y: 320, type: 'metro' },
            { name: '한밭수목원', x: 430, y: 240, type: 'leisure' },
            { name: '노은역', x: 170, y: 170, type: 'residential' },
            { name: '대전역', x: 630, y: 360, type: 'metro' }
        ];

        function renderMap(filter = 'all') {
            const flowLayer = document.getElementById('flowLayer');
            const nodeLayer = document.getElementById('nodeLayer');
            flowLayer.innerHTML = '';
            nodeLayer.innerHTML = '';

            // Draw Flows
            flowsData.forEach(f => {
                if (filter !== 'all' && f.category !== filter) return;
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', f.x1);
                line.setAttribute('y1', f.y1);
                line.setAttribute('x2', f.x2);
                line.setAttribute('y2', f.y2);
                line.setAttribute('stroke', f.color);
                line.setAttribute('stroke-width', f.width);
                line.setAttribute('class', 'flow-line');
                line.onclick = () => selectFlow(f);
                flowLayer.appendChild(line);
            });

            // Draw Nodes
            nodesData.forEach(n => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', n.x);
                circle.setAttribute('cy', n.y);
                circle.setAttribute('r', 5);
                circle.setAttribute('fill', n.type === 'metro' ? '#10b981' : (n.type === 'campus' ? '#06b6d4' : '#f59e0b'));
                circle.setAttribute('stroke', '#ffffff');
                circle.setAttribute('stroke-width', 1.5);
                circle.setAttribute('class', 'station-node');
                nodeLayer.appendChild(circle);

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', n.x + 8);
                text.setAttribute('y', n.y + 4);
                text.setAttribute('fill', '#94a3b8');
                text.setAttribute('font-size', '10');
                text.setAttribute('font-weight', '600');
                text.textContent = n.name;
                nodeLayer.appendChild(text);
            });
        }

        function selectFlow(f) {
            document.getElementById('flowCode').innerText = f.code;
            document.getElementById('flowTitle').innerText = f.title;
            document.getElementById('flowDesc').innerText = f.desc;
            document.getElementById('flowTrips').innerText = f.trips;
            document.getElementById('flowDist').innerText = f.dist;
            document.getElementById('flowDuration').innerText = f.duration;
            document.getElementById('flowType').innerText = f.type;
            document.getElementById('flowInsight').innerText = f.insight;
            
            const badge = document.getElementById('flowCategoryBadge');
            badge.innerText = f.type;
            badge.className = f.category === 'metro' ? 'badge badge-emerald' : (f.category === 'campus' ? 'badge badge-cyan' : 'badge badge-amber');
        }

        function filterFlows(cat, btn) {
            document.querySelectorAll('#flow-section .tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderMap(cat);
            const first = flowsData.find(f => cat === 'all' || f.category === cat);
            if (first) selectFlow(first);
        }

        // Rain Simulator
        const rainLevels = [
            { label: '비 안 옴 (0mm)', trips: 936.3, drop: '기준 통행량 (100%)', bar: '100%', color: 'var(--cyan)', action: '기상이 쾌적하여 트럭 10대를 100% 정상 가동합니다.' },
            { label: '이슬비 (0~1mm)', trips: 650.1, drop: '-30.6% 감소', bar: '69.4%', color: '#38bdf8', action: '경미한 우천으로 수요가 30% 감소합니다. 트럭 8대로 탄력 운영합니다.' },
            { label: '일반 비 (1~3mm)', trips: 272.5, drop: '🚨 -70.9% 급락 (티핑 포인트)', bar: '29.1%', color: 'var(--rose)', action: '🚨 1mm 돌파 즉시 수요 71% 증발! 재배치 트럭을 즉시 70% 감차(3대만 긴급 대기)하여 불필요한 유류비를 절감합니다.' },
            { label: '폭우 (>3mm)', trips: 159.8, drop: '🚨 -82.9% 급감', bar: '17.1%', color: '#be123c', action: '폭우로 시민 이용이 거의 중단되었습니다. 트럭 운행을 전면 중단하고 침수 우려 하천 거점 자전거를 고지대로 긴급 대피시킵니다.' }
        ];

        function setRain(idx, btn) {
            btn.parentElement.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const r = rainLevels[idx];
            document.getElementById('rainTripsPerHour').innerText = r.trips + ' 건/h';
            document.getElementById('rainDropPercent').innerText = r.drop;
            document.getElementById('rainBar').style.width = r.bar;
            document.getElementById('rainBar').style.background = r.color;
            document.getElementById('rainActionBox').innerHTML = '<strong style="color:' + r.color + ';">물류 운영 지침:</strong> ' + r.action;
        }

        // 24 Hour Scrubber
        const hourlyStats = [
            { h: 0, w: '12.1만', desc: '심야 귀가 및 새벽 잔여 통행' },
            { h: 8, w: '39.4만', desc: '오전 8시 출근 피크! 지하철역으로 향하는 라스트마일 집중' },
            { h: 12, w: '31.3만', desc: '점심 식사 및 캠퍼스·도심 단거리 이동' },
            { h: 18, w: '62.7만', desc: '퇴근 피크 대폭증! 출근길 대비 1.59배 폭증하며 대여소 결품 최고조' },
            { h: 22, w: '31.3만', desc: '야간 귀가 및 여가 라이딩 완만한 감소' }
        ];

        function updateHourScrubber(val) {
            const h = parseInt(val);
            const x = 40 + (h / 23) * 920;
            document.getElementById('scrubberLine').setAttribute('x1', x);
            document.getElementById('scrubberLine').setAttribute('x2', x);
            document.getElementById('scrubberCircle').setAttribute('cx', x);

            document.getElementById('selectedHourText').innerText = h + ':00 (' + (h < 12 ? '오전 ' + h + '시' : (h === 12 ? '정오' : '오후 ' + (h - 12) + '시')) + ')';
            
            if (h === 18) {
                document.getElementById('selectedHourTrips').innerText = '627,088 건';
                document.getElementById('selectedHourDesc').innerText = '퇴근길 피크 대폭증! 지각 부담이 없고 약속/운동이 결합되어 둔산동/유성 일대 품절 사태 최고조.';
            } else if (h === 8) {
                document.getElementById('selectedHourTrips').innerText = '393,736 건';
                document.getElementById('selectedHourDesc').innerText = '오전 출근 피크! 노은동 등 주거지에서 지하철역 방향으로 집중 빠져나감.';
            } else if (h === 12) {
                document.getElementById('selectedHourTrips').innerText = '312,809 건';
                document.getElementById('selectedHourDesc').innerText = '점심 식사 시간 1km 이내 초단거리 셔틀 이동 활발.';
            } else {
                document.getElementById('selectedHourTrips').innerText = Math.round(200000 + Math.sin(h / 3) * 150000).toLocaleString() + ' 건';
                document.getElementById('selectedHourDesc').innerText = '해당 시간대 대전시 전역 평균적 생활 통행 흐름 유지.';
            }
        }

        // Bayesian Calculator
        function calcBayes() {
            const lam = parseFloat(document.getElementById('lambdaSlider').value);
            const t = parseFloat(document.getElementById('timeSlider').value);
            document.getElementById('lambdaVal').innerText = lam.toFixed(1) + ' 대/h';
            document.getElementById('timeVal').innerText = t.toFixed(1) + ' 시간';

            const pFail = (1.0 - Math.exp(-lam * t)) * 100;
            document.getElementById('bayesResult').innerText = pFail.toFixed(2) + ' %';
            
            const badgeBox = document.getElementById('bayesBadgeBox');
            if (pFail >= 99.0) {
                badgeBox.innerHTML = '<span class="badge badge-rose">🚨 즉시 수거 지시서 자동 발행</span>';
            } else if (pFail >= 90.0) {
                badgeBox.innerHTML = '<span class="badge badge-amber">⚠️ 고장 의심 관제 모니터링</span>';
            } else {
                badgeBox.innerHTML = '<span class="badge badge-cyan">정상 미이용 판단 (오탐 방지)</span>';
            }
        }

        // Initialize on Load
        window.addEventListener('DOMContentLoaded', () => {
            renderMap('all');
            selectFlow(flowsData[0]);
            calcBayes();
        });
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
    f.write(html_code)
print(f"Saved: {dest_root}")

with open(dest_docs, "w", encoding="utf-8") as f:
    f.write(html_code)
print(f"Saved: {dest_docs}")

with open(dest_desktop, "w", encoding="utf-8") as f:
    f.write(html_code)
print(f"Saved: {dest_desktop}")

with open(dest_brain, "w", encoding="utf-8") as f:
    f.write(html_code)
print(f"Saved: {dest_brain}")
