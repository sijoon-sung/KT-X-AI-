import json
import os
from pathlib import Path

# Load EDA summary
ROOT = Path("C:/Users/sijoo/Documents/tashu")
with open(ROOT / "outputs/eda_analysis_summary.json", "r", encoding="utf-8") as f:
    eda = json.load(f)

html_content = f"""<!DOCTYPE html>
<html lang="ko" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KT-X-AI | 대전 공영자전거 '타슈' 300m Super-Station AI 재배치 & 스마트 관제 시스템</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Google Fonts Pretendard & Inter -->
    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        brand: {{
                            50: '#ecfeff',
                            100: '#cffafe',
                            400: '#22d3ee',
                            500: '#06b6d4',
                            600: '#0891b2',
                            900: '#164e63',
                        }},
                        dark: {{
                            bg: '#0a0f1d',
                            card: '#111827',
                            border: '#1f2937',
                            surface: '#1e293b'
                        }}
                    }},
                    fontFamily: {{
                        sans: ['Pretendard', 'Inter', '-apple-system', 'sans-serif'],
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{
            font-family: 'Pretendard', sans-serif;
            background-color: #0a0f1d;
            color: #f3f4f6;
        }}
        .glass-panel {{
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .glass-panel-glow {{
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(6, 182, 212, 0.3);
            box-shadow: 0 0 25px rgba(6, 182, 212, 0.15);
        }}
        .custom-scrollbar::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb {{
            background: #374151;
            border-radius: 4px;
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {{
            background: #4b5563;
        }}
    </style>
</head>
<body class="bg-[#0a0f1d] text-slate-100 min-h-screen selection:bg-cyan-500 selection:text-white">

    <!-- Top Navigation -->
    <header class="fixed top-0 left-0 right-0 z-50 glass-panel border-b border-slate-800">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 font-black text-slate-950 text-lg">
                    T
                </div>
                <div>
                    <span class="font-extrabold text-base tracking-tight text-white flex items-center gap-2">
                        KT-X-AI <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 font-mono">Tashu AI 2.0</span>
                    </span>
                    <p class="text-[10px] text-slate-400 hidden sm:block">대전 공영자전거 300m Super-Station AI 재배치 & 스마트 관제</p>
                </div>
            </div>
            
            <nav class="hidden md:flex items-center space-x-1 lg:space-x-2 text-xs font-medium text-slate-300">
                <a href="#overview" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">개요 & 아키텍처</a>
                <a href="#eda" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">911만 건 EDA</a>
                <a href="#ai-model" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">300m Q85 모델</a>
                <a href="#routing" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">트럭 10대 동선</a>
                <a href="#bayesian" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">베이지안 관제</a>
                <a href="#synergy" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">도시교통 & ESG</a>
                <a href="#code" class="px-3 py-1.5 rounded-lg hover:text-white hover:bg-slate-800/60 transition">코드 & 리포트</a>
            </nav>

            <div class="flex items-center space-x-3">
                <a href="https://github.com/sijoon-sung/KT-X-AI-/tree/성시준" target="_blank" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition">
                    <svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                    <span>Branch: 성시준</span>
                </a>
            </div>
        </div>
    </header>

    <main class="pt-24 pb-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-24">
        
        <!-- HERO SECTION -->
        <section class="relative pt-8 pb-12 text-center lg:text-left">
            <div class="absolute -top-12 left-1/2 -translate-x-1/2 w-3/4 h-64 bg-gradient-to-r from-cyan-500/10 via-emerald-500/10 to-indigo-500/10 blur-3xl -z-10 rounded-full"></div>
            
            <div class="flex flex-col lg:flex-row items-center justify-between gap-12">
                <div class="flex-1 space-y-6">
                    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-700/50 text-cyan-400 text-xs font-semibold">
                        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        KT-X-AI 빅데이터 & 인공지능 실증 프로젝트
                    </div>
                    <h1 class="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-tight text-white">
                        대전 공영자전거 <span class="bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">타슈(Tashu)</span><br/>
                        AI 재배치 & 스마트 관제
                    </h1>
                    <p class="text-sm sm:text-base text-slate-300 leading-relaxed max-w-2xl font-light">
                        <strong class="text-cyan-300 font-semibold">911만 건 전수 통행 빅데이터</strong>를 심층 분석하여 
                        출퇴근 품절 사태를 해결하는 <strong class="text-emerald-300 font-semibold">300m Super-Station Q85 머신러닝</strong>, 
                        퇴근 60분 골든타임을 사수하는 <strong class="text-cyan-300 font-semibold">물리 제약 트럭 10대 동선 최적화</strong>, 
                        그리고 API 기반 <strong class="text-amber-300 font-semibold">베이지안 고장 유령 자전거 감지</strong>를 결합한 차세대 공공 모빌리티 종합 솔루션입니다.
                    </p>
                    <div class="flex flex-wrap items-center justify-center lg:justify-start gap-4 pt-2">
                        <a href="#eda" class="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 hover:opacity-95 transition flex items-center gap-2">
                            <span>911만 건 데이터 분석 리포트 보기</span>
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                        </a>
                        <a href="#ai-model" class="px-6 py-3 rounded-xl glass-panel hover:bg-slate-800 text-white font-semibold text-sm border border-slate-700 transition">
                            AI 재배치 모델 & 검증
                        </a>
                    </div>
                </div>

                <!-- 5 Key Highlights Stats Grid -->
                <div class="grid grid-cols-2 gap-4 w-full lg:w-auto shrink-0 max-w-md">
                    <div class="glass-panel p-5 rounded-2xl border-cyan-500/30 space-y-1">
                        <span class="text-xs text-slate-400 font-medium">분석 통행 데이터</span>
                        <div class="text-2xl sm:text-3xl font-black text-cyan-400">9,116,462<span class="text-sm font-normal text-slate-400 ml-1">건</span></div>
                        <p class="text-[11px] text-slate-400">1개년 전수 로그 + 시간별 기상 결합</p>
                    </div>
                    <div class="glass-panel p-5 rounded-2xl border-emerald-500/30 space-y-1">
                        <span class="text-xs text-slate-400 font-medium">생활권 Super-Station</span>
                        <div class="text-2xl sm:text-3xl font-black text-emerald-400">460<span class="text-sm font-normal text-slate-400 ml-1">개 거점</span></div>
                        <p class="text-[11px] text-slate-400">1,200개 대여소 300m 보행권 압축</p>
                    </div>
                    <div class="glass-panel p-5 rounded-2xl border-amber-500/30 space-y-1">
                        <span class="text-xs text-slate-400 font-medium">피크 결품 위험 차단</span>
                        <div class="text-2xl sm:text-3xl font-black text-amber-400">-65.0<span class="text-sm font-normal text-slate-400 ml-1">%</span></div>
                        <p class="text-[11px] text-slate-400">Top 20 결품률 38% → 13.3% 급감</p>
                    </div>
                    <div class="glass-panel p-5 rounded-2xl border-indigo-500/30 space-y-1">
                        <span class="text-xs text-slate-400 font-medium">트럭 10대 작업 완료</span>
                        <div class="text-2xl sm:text-3xl font-black text-indigo-400">48.5<span class="text-sm font-normal text-slate-400 ml-1">분 / 176대</span></div>
                        <p class="text-[11px] text-slate-400">퇴근 60분 골든타임 이내 완벽 하차</p>
                    </div>
                    <div class="col-span-2 glass-panel p-4 rounded-2xl border-emerald-500/40 flex items-center justify-between">
                        <div>
                            <span class="text-xs text-slate-400 font-medium">ESG 이산화탄소 순감축 증빙</span>
                            <div class="text-xl font-black text-emerald-300">701.8 톤 CO₂ <span class="text-xs font-normal text-slate-400">(소나무 10.6만 그루)</span></div>
                        </div>
                        <div class="text-right">
                            <span class="text-xs text-slate-400 font-medium">시민 유류비 절감</span>
                            <div class="text-lg font-bold text-slate-200">약 6.89억 원</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 1: OVERVIEW & SYSTEM ARCHITECTURE -->
        <section id="overview" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-cyan-400 uppercase tracking-widest">Section 01</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">문제 정의 & 엔드투엔드 시스템 아키텍처</h2>
                <p class="text-sm text-slate-400 mt-1">왜 기존의 대여소 단위 예측이 실패했는가? 데이터와 물리적 제약을 극복한 6단계 파이프라인</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Problem Card -->
                <div class="glass-panel p-6 rounded-2xl border-red-500/30 space-y-4">
                    <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-lg bg-red-950/80 border border-red-800 text-red-400 flex items-center justify-center font-bold">!</div>
                        <h3 class="text-lg font-bold text-white">현장 운영의 치명적 병목 (Pain Points)</h3>
                    </div>
                    <ul class="space-y-3 text-xs sm:text-sm text-slate-300 leading-relaxed">
                        <li class="flex items-start gap-2">
                            <span class="text-red-400 mt-0.5">•</span>
                            <span><strong>출퇴근 극단적 쏠림:</strong> 출근 시 노은동·궁동 등 주거지 결품(0대) 발생, 둔산동 정부청사 거점은 자전거 포화(Surplus)로 반납 불가.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-red-400 mt-0.5">•</span>
                            <span><strong>개별 대여소의 시계열 희소성:</strong> 1,200개 대여소가 50~100m 간격으로 쪼개져 있어, 개별 예측 시 0건 데이터가 태반(Sparsity 문제).</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-red-400 mt-0.5">•</span>
                            <span><strong>탁상공론식 트럭 순회:</strong> 트럭 10대가 7~8개 대여소를 찔끔찔끔 돌다 퇴근 골든타임(17~18시)을 놓치고 공차 주행 증가.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-red-400 mt-0.5">•</span>
                            <span><strong>고장 유령 자전거 방치:</strong> 시민 민원 접수 전까지는 1대 고착 자전거가 앱에 대여 가능으로 떠서 헛걸음 유발.</span>
                        </li>
                    </ul>
                </div>

                <!-- Solution Card -->
                <div class="glass-panel p-6 rounded-2xl border-cyan-500/30 space-y-4">
                    <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-800 text-cyan-400 flex items-center justify-center font-bold">✓</div>
                        <h3 class="text-lg font-bold text-white">차세대 AI 솔루션 혁신 (Key Breakthroughs)</h3>
                    </div>
                    <ul class="space-y-3 text-xs sm:text-sm text-slate-300 leading-relaxed">
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 mt-0.5">•</span>
                            <span><strong>300m 생활권 Super-Station (460개):</strong> 도보 3분 이내 대체 가능 대여소를 통합하여 데이터 밀도 확보 및 도보 환승 유도.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 mt-0.5">•</span>
                            <span><strong>Quantile 85% Asymmetric Loss:</strong> MSE 대신 부족 오차에 5.67배 높은 벌점을 주어 결품을 선제 방어하는 안전 버퍼 구축.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 mt-0.5">•</span>
                            <span><strong>[상차 1곳 → 하차 1~2곳] 물리 제약 Fleet Routing:</strong> 상하차 시간 및 시내 퇴근길 22km/h 속도를 반영해 48.5분 만에 176대 하차.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 mt-0.5">•</span>
                            <span><strong>포아송 베이지안 추론:</strong> 자전거 ID 없이도 시간당 수요 대비 고착 시간을 계산해 고장 자전거를 99.9% 확률로 적발.</span>
                        </li>
                    </ul>
                </div>
            </div>

            <!-- Pipeline Diagram -->
            <div class="glass-panel p-6 rounded-2xl">
                <h4 class="text-sm font-semibold text-slate-300 mb-6 flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                    데이터 수집부터 현장 배차까지의 엔드투엔드 파이프라인
                </h4>
                <div class="grid grid-cols-1 md:grid-cols-6 gap-3 text-center text-xs">
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="text-[10px] text-cyan-400 font-mono font-bold">STEP 01</div>
                        <div class="font-bold text-slate-200">데이터 수집·정제</div>
                        <p class="text-[11px] text-slate-400 leading-tight">911만 건 통행 로그 + 기상청 기상 + OSM 도로망 전처리</p>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="text-[10px] text-emerald-400 font-mono font-bold">STEP 02</div>
                        <div class="font-bold text-slate-200">300m 군집화</div>
                        <p class="text-[11px] text-slate-400 leading-tight">1,200개 대여소 → 460개 생활권 Super-Station 집약</p>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="text-[10px] text-cyan-400 font-mono font-bold">STEP 03</div>
                        <div class="font-bold text-slate-200">OD 네트워크 피처</div>
                        <p class="text-[11px] text-slate-400 leading-tight">87.4% 타 군집 유입량(`lag1_ret`) 주입 (중요도 53.4%)</p>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="text-[10px] text-amber-400 font-mono font-bold">STEP 04</div>
                        <div class="font-bold text-slate-200">XGBoost Q85 예측</div>
                        <p class="text-[11px] text-slate-400 leading-tight">결품 위험에 5.67배 벌점 부여, 안전 재고 자동 산출</p>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="text-[10px] text-indigo-400 font-mono font-bold">STEP 05</div>
                        <div class="font-bold text-slate-200">트럭 동선 최적화</div>
                        <p class="text-[11px] text-slate-400 leading-tight">10대 트럭 물리 동선 설계, 48.5분 만에 176대 공급</p>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="text-[10px] text-rose-400 font-mono font-bold">STEP 06</div>
                        <div class="font-bold text-slate-200">베이지안 관제</div>
                        <p class="text-[11px] text-slate-400 leading-tight">API 스냅샷 역산으로 1대 고착 자전거 99.9% 적발</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 2: COMPREHENSIVE 9.11M EDA -->
        <section id="eda" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Section 02</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">911만 건 시민 통행 빅데이터 심층 EDA</h2>
                <p class="text-sm text-slate-400 mt-1">데이터 분석가가 밝혀낸 1km의 법칙, 출퇴근 쌍봉 리듬, 1mm 강수 임계점, 그리고 최상위 이동 회랑</p>
            </div>

            <!-- 4 Visual Cards Grid -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                <!-- Chart 1: Distance Distribution -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <div class="flex items-center justify-between">
                        <div>
                            <h3 class="text-base font-bold text-white">① 통행 거리 및 시간: "1km & 11분의 법칙"</h3>
                            <p class="text-xs text-slate-400">전체 통행의 75.5%가 2km 이내에서 완결</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 text-xs font-mono font-bold">Median: 1.0 km / 11분</span>
                    </div>
                    <div class="h-64 flex items-center justify-center">
                        <canvas id="distanceChart"></canvas>
                    </div>
                    <div class="bg-slate-900/70 p-3 rounded-xl border border-slate-800 text-xs text-slate-300 leading-relaxed">
                        <strong class="text-cyan-300">인사이트:</strong> 51.2%가 1km 이내 초단거리이며, 75.5%가 2km 이내입니다. 타슈는 도시 횡단용이 아닌 <strong>도보 10~15분 거리를 3~5분으로 줄여주는 라스트마일 마이크로 모빌리티</strong>로 정착되어 있습니다.
                    </div>
                </div>

                <!-- Chart 2: Diurnal Rhythm (M-Curve vs Bell-Curve) -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <div class="flex items-center justify-between">
                        <div>
                            <h3 class="text-base font-bold text-white">② 24시간 통행 리듬: "평일 쌍봉 vs 주말 종형"</h3>
                            <p class="text-xs text-slate-400">퇴근길(18시 62.7만 건)은 출근길(08시 39.4만 건)의 1.59배</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs font-mono font-bold">18시 피크: 62.7만 건</span>
                    </div>
                    <div class="h-64">
                        <canvas id="hourlyChart"></canvas>
                    </div>
                    <div class="bg-slate-900/70 p-3 rounded-xl border border-slate-800 text-xs text-slate-300 leading-relaxed">
                        <strong class="text-emerald-300">인사이트:</strong> 평일은 전형적인 통근형 M자 쌍봉을 보이며, 퇴근 시간이 출근보다 1.6배 큽니다(여유 있는 귀가, 저녁 약속, 운동 결합). 반면 주말은 오후 14~17시 완만한 종형 곡선으로 평균 이용시간이 28.4분으로 증가합니다.
                    </div>
                </div>

                <!-- Chart 3: Weather Sensitivity -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <div class="flex items-center justify-between">
                        <div>
                            <h3 class="text-base font-bold text-white">③ 날씨 탄력성: "1mm 강수 급락 임계선"</h3>
                            <p class="text-xs text-slate-400">주간(07~22시) 시간당 통행량 변화 (비 & 기온)</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-amber-950 text-amber-400 border border-amber-800 text-xs font-mono font-bold">1mm 강수 시 -70.9%</span>
                    </div>
                    <div class="h-64">
                        <canvas id="weatherChart"></canvas>
                    </div>
                    <div class="bg-slate-900/70 p-3 rounded-xl border border-slate-800 text-xs text-slate-300 leading-relaxed">
                        <strong class="text-amber-300">운영 룰 제안:</strong> 강수량 1mm 돌파 시 수요의 71%가 증발합니다. 기상 예보 상 1mm 이상 강수 감지 시 <strong>재배치 트럭 운행을 즉시 70% 감차</strong>하여 불필요한 유류비와 인건비를 절감해야 합니다.
                    </div>
                </div>

                <!-- Card 4: Top 3 OD Flow Corridors -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <div class="flex items-center justify-between">
                        <div>
                            <h3 class="text-base font-bold text-white">④ 최상위 통행 회랑 (Top Corridors) 3대 유형</h3>
                            <p class="text-xs text-slate-400">911만 건 통행 중 가장 밀집된 O-D 패턴</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-indigo-950 text-indigo-400 border border-indigo-800 text-xs font-mono font-bold">Top 3 Clusters</span>
                    </div>
                    
                    <div class="space-y-3 pt-1">
                        <div class="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 flex items-center justify-between">
                            <div class="space-y-1">
                                <div class="flex items-center gap-2">
                                    <span class="px-2 py-0.5 rounded bg-cyan-900/50 text-cyan-300 text-[10px] font-bold">유형 A: 캠퍼스 셔틀</span>
                                    <span class="text-xs font-bold text-slate-200">KAIST 학사식당 ⇄ 창의학습관 / 서쪽 쪽문</span>
                                </div>
                                <p class="text-[11px] text-slate-400">평균 0.47~0.65km | 6.8~8.9분 소요 | 수업 및 식사 이동</p>
                            </div>
                            <div class="text-right">
                                <div class="text-sm font-extrabold text-cyan-400">34,984<span class="text-xs font-normal text-slate-400">건</span></div>
                                <span class="text-[10px] text-slate-500">연간 이동량</span>
                            </div>
                        </div>

                        <div class="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 flex items-center justify-between">
                            <div class="space-y-1">
                                <div class="flex items-center gap-2">
                                    <span class="px-2 py-0.5 rounded bg-emerald-900/50 text-emerald-300 text-[10px] font-bold">유형 B: 지하철 환승</span>
                                    <span class="text-xs font-bold text-slate-200">정부청사역 4번출구 → 정부청사 입구(남문)</span>
                                </div>
                                <p class="text-[11px] text-slate-400">평균 0.38km | 3.1분 소요 | 지하철 1호선 직결 출근 통행</p>
                            </div>
                            <div class="text-right">
                                <div class="text-sm font-extrabold text-emerald-400">4,964<span class="text-xs font-normal text-slate-400">건</span></div>
                                <span class="text-[10px] text-slate-500">단일 방향</span>
                            </div>
                        </div>

                        <div class="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 flex items-center justify-between">
                            <div class="space-y-1">
                                <div class="flex items-center gap-2">
                                    <span class="px-2 py-0.5 rounded bg-amber-900/50 text-amber-300 text-[10px] font-bold">유형 C: 공원 레저</span>
                                    <span class="text-xs font-bold text-slate-200">한밭수목원(서원) ⇄ 한밭수목원(동원)</span>
                                </div>
                                <p class="text-[11px] text-slate-400">평균 0.65~1.22km | 30.7분 체류 | 여가 및 힐링 루프</p>
                            </div>
                            <div class="text-right">
                                <div class="text-sm font-extrabold text-amber-400">10,054<span class="text-xs font-normal text-slate-400">건</span></div>
                                <span class="text-[10px] text-slate-500">주말 집중</span>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </section>

        <!-- SECTION 3: AI DEMAND PREDICTION & 300M REBALANCING -->
        <section id="ai-model" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-cyan-400 uppercase tracking-widest">Section 03</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">300m Super-Station & Quantile 85% 비대칭 AI 모델링</h2>
                <p class="text-sm text-slate-400 mt-1">결품 오차에 5.67배 벌점을 부여하여 품절 사태를 65% 차단한 머신러닝 아키텍처</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Concept 1 -->
                <div class="glass-panel p-6 rounded-2xl space-y-3">
                    <div class="w-10 h-10 rounded-xl bg-cyan-950 border border-cyan-800 text-cyan-400 flex items-center justify-center font-bold">300m</div>
                    <h3 class="text-base font-bold text-white">공간 압축: 460개 생활권 Super-Station</h3>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        대전 시내 1,200개 대여소 중 상당수는 50~150m 거리에 흩어져 있어 각 대여소별 통행 시계열이 심각하게 쪼개져 있었습니다.
                        <strong>보행 3분 반경(300m) 내 앵커 거점</strong>으로 대여소를 통합하여 460개 Super-Station으로 묶음으로써 데이터 밀도를 극대화하고 거점 간 도보 대체성을 확보했습니다.
                    </p>
                </div>

                <!-- Concept 2 -->
                <div class="glass-panel p-6 rounded-2xl space-y-3">
                    <div class="w-10 h-10 rounded-xl bg-emerald-950 border border-emerald-800 text-emerald-400 flex items-center justify-center font-bold">OD</div>
                    <h3 class="text-base font-bold text-white">87.4% 네트워크 유입 회랑 피처 주입</h3>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        전체 통행의 87.4%는 다른 군집에서 들어오는 통행입니다. 대여소 자체의 시계열 래그만 보는 기존 방식을 탈피하여,
                        <strong>직전 시간대(t-1)에 해당 거점으로 향해 출발한 통행량(`lag1_ret`)</strong>을 네트워크 피처로 주입했습니다. 이 피처는 모델 중요도 <strong>53.41%</strong>를 차지하며 예측력을 대폭 끌어올렸습니다.
                    </p>
                </div>

                <!-- Concept 3 -->
                <div class="glass-panel p-6 rounded-2xl space-y-3">
                    <div class="w-10 h-10 rounded-xl bg-amber-950 border border-amber-800 text-amber-400 flex items-center justify-center font-bold">Q85</div>
                    <h3 class="text-base font-bold text-white">Quantile 85% Asymmetric Loss (τ=0.85)</h3>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        전통적인 MSE(평균제곱오차)는 자전거가 +2대 남는 것과 -2대 모자란 것을 똑같이 취급합니다.
                        하지만 공공 모빌리티에서는 <strong>모자라는 것(결품)이 시민 불만을 야기하는 치명적 오류</strong>입니다.
                        Pinball Loss(τ=0.85)를 적용하여 부족 오차에 <strong>5.67배 높은 페널티</strong>를 부여해 피크 수요 버퍼를 자동 산출합니다.
                    </p>
                </div>
            </div>

            <!-- Before vs After Metrics Bar -->
            <div class="glass-panel-glow p-6 rounded-2xl">
                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div class="space-y-2">
                        <span class="text-xs font-bold text-cyan-400 uppercase tracking-widest">Model Validation Results</span>
                        <h3 class="text-xl font-bold text-white">실제 현장 개선 효과 검증 (Before vs After)</h3>
                        <p class="text-xs text-slate-300 max-w-xl">
                            2025년 9월 15일 17:00 퇴근 피크 시뮬레이션 결과, 결품 고위험 거점 Top 20의 품절 방치율이 38.0%에서 13.3%로 65% 급감했습니다.
                        </p>
                    </div>

                    <div class="flex items-center gap-6 shrink-0">
                        <div class="text-center">
                            <span class="text-xs text-slate-400">기존 MSE 모델 품절률</span>
                            <div class="text-3xl font-black text-rose-400 line-through">38.0%</div>
                            <span class="text-[10px] text-slate-500">10회 중 4회 0대 방치</span>
                        </div>
                        <div class="text-2xl font-bold text-slate-600">→</div>
                        <div class="text-center">
                            <span class="text-xs text-cyan-300 font-semibold">Q85 슈퍼스테이션 적용 후</span>
                            <div class="text-4xl font-black text-emerald-400">13.3%</div>
                            <span class="text-[10px] text-emerald-300 font-medium">결품 위험 65% 차단</span>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6 pt-6 border-t border-slate-800 text-xs">
                    <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
                        <span class="text-slate-400">퇴근 1시간 즉시 구제 인원</span>
                        <div class="text-xl font-bold text-cyan-400 mt-1">73명 / 1시간</div>
                        <p class="text-[11px] text-slate-500 mt-0.5">월간 4,500명 헛걸음 방지</p>
                    </div>
                    <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
                        <span class="text-slate-400">집중 투하 5대 거점 결품 방어율</span>
                        <div class="text-xl font-bold text-emerald-400 mt-1">88.8% 방어</div>
                        <p class="text-[11px] text-slate-500 mt-0.5">82명 결품 → 9명으로 최소화</p>
                    </div>
                    <div class="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
                        <span class="text-slate-400">투입 자전거 유효 회전율</span>
                        <div class="text-xl font-bold text-amber-400 mt-1">30.2% 회전</div>
                        <p class="text-[11px] text-slate-500 mt-0.5">배차 후 1시간 내 즉시 대여</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 4: REALISTIC FLEET ROUTING -->
        <section id="routing" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-indigo-400 uppercase tracking-widest">Section 04</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">물리 제약 반영 트럭 10대 동선 최적화 (Fleet Routing)</h2>
                <p class="text-sm text-slate-400 mt-1">탁상공론식 7개 대여소 순회를 타파하고, 상하차 시간과 퇴근길 시내 속도를 반영한 [상차 1곳 → 하차 1~2곳] 핀포인트 릴레이</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <!-- Constraints Overview -->
                <div class="glass-panel p-6 rounded-2xl space-y-4 lg:col-span-1">
                    <h3 class="text-base font-bold text-white">현장 물리 제약 모델</h3>
                    <ul class="space-y-3 text-xs text-slate-300">
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 font-bold">1.</span>
                            <span><strong>정차 준비:</strong> 거점당 3분 (주차, 윙바디/리프트 개방, 단말기 체크)</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 font-bold">2.</span>
                            <span><strong>상하차 속도:</strong> 상차 1대당 45초, 하차 1대당 30초 실측 반영</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 font-bold">3.</span>
                            <span><strong>도로 주행:</strong> 퇴근길 시내 22km/h, 굴절률 1.3배 적용</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-cyan-400 font-bold">4.</span>
                            <span><strong>배차 구조:</strong> [1곳에서 18대 만차 상차 → 결품 거점 1~2곳 전량 하차]</span>
                        </li>
                    </ul>
                    <div class="p-3 bg-indigo-950/40 rounded-xl border border-indigo-800/40 text-[11px] text-indigo-200">
                        <strong>성과:</strong> 평균 7.5km 주행, 48.5분 만에 176대 하차를 완료하여 퇴근 60분 골든타임 이내 임무 완수.
                    </div>
                </div>

                <!-- Truck Manifest Table -->
                <div class="glass-panel p-6 rounded-2xl lg:col-span-3 space-y-4">
                    <div class="flex items-center justify-between">
                        <h3 class="text-base font-bold text-white">퇴근 피크(17:00) 트럭 10대 실전 배차 지시서 (Dispatch Manifest)</h3>
                        <span class="text-xs text-slate-400 font-mono">총 176대 재배치 | 평균 48.5분</span>
                    </div>

                    <div class="overflow-x-auto custom-scrollbar">
                        <table class="w-full text-left text-xs border-collapse">
                            <thead>
                                <tr class="border-b border-slate-800 text-slate-400 uppercase font-mono text-[11px]">
                                    <th class="py-2.5 px-3">트럭</th>
                                    <th class="py-2.5 px-3">상차 거점 (잉여 거점)</th>
                                    <th class="py-2.5 px-3">하차 거점 (결품 거점)</th>
                                    <th class="py-2.5 px-3 text-center">수량</th>
                                    <th class="py-2.5 px-3 text-right">주행거리</th>
                                    <th class="py-2.5 px-3 text-right">총 소요시간</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800/60 font-sans text-slate-200">
                                <tr class="hover:bg-slate-800/30">
                                    <td class="py-2.5 px-3 font-bold text-cyan-400">트럭 01</td>
                                    <td class="py-2.5 px-3">노은역 3번 출구 (잉여 45대)</td>
                                    <td class="py-2.5 px-3">유성온천역 7번 출구 (결품 18대)</td>
                                    <td class="py-2.5 px-3 text-center font-bold text-emerald-400">18대</td>
                                    <td class="py-2.5 px-3 text-right text-slate-300">6.8 km</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-cyan-300">46.2 분</td>
                                </tr>
                                <tr class="hover:bg-slate-800/30">
                                    <td class="py-2.5 px-3 font-bold text-cyan-400">트럭 02</td>
                                    <td class="py-2.5 px-3">궁동 로데오거리 남측 (잉여 38대)</td>
                                    <td class="py-2.5 px-3">월평역 1번 출구 (결품 16대)</td>
                                    <td class="py-2.5 px-3 text-center font-bold text-emerald-400">16대</td>
                                    <td class="py-2.5 px-3 text-right text-slate-300">5.4 km</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-cyan-300">42.8 분</td>
                                </tr>
                                <tr class="hover:bg-slate-800/30">
                                    <td class="py-2.5 px-3 font-bold text-cyan-400">트럭 03</td>
                                    <td class="py-2.5 px-3">만년동 KBS 방송국 (잉여 32대)</td>
                                    <td class="py-2.5 px-3">정부청사역 2번 출구 (결품 18대)</td>
                                    <td class="py-2.5 px-3 text-center font-bold text-emerald-400">18대</td>
                                    <td class="py-2.5 px-3 text-right text-slate-300">4.9 km</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-cyan-300">41.5 분</td>
                                </tr>
                                <tr class="hover:bg-slate-800/30">
                                    <td class="py-2.5 px-3 font-bold text-cyan-400">트럭 04</td>
                                    <td class="py-2.5 px-3">도안 신도시 수변공원 (잉여 29대)</td>
                                    <td class="py-2.5 px-3">원신흥동 주민센터 (결품 18대)</td>
                                    <td class="py-2.5 px-3 text-center font-bold text-emerald-400">18대</td>
                                    <td class="py-2.5 px-3 text-right text-slate-300">7.2 km</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-cyan-300">47.8 분</td>
                                </tr>
                                <tr class="hover:bg-slate-800/30">
                                    <td class="py-2.5 px-3 font-bold text-cyan-400">트럭 05</td>
                                    <td class="py-2.5 px-3">탄방동 남선공원 체육관 (잉여 27대)</td>
                                    <td class="py-2.5 px-3">시청역 8번 출구 (결품 18대)</td>
                                    <td class="py-2.5 px-3 text-center font-bold text-emerald-400">18대</td>
                                    <td class="py-2.5 px-3 text-right text-slate-300">6.1 km</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-cyan-300">45.0 분</td>
                                </tr>
                                <tr class="hover:bg-slate-800/30">
                                    <td class="py-2.5 px-3 font-bold text-cyan-400">트럭 06~10</td>
                                    <td class="py-2.5 px-3">관저동/가오동/송촌동 주요 잉여 거점</td>
                                    <td class="py-2.5 px-3">대동역/서대전네거리역 등 주요 결품 거점</td>
                                    <td class="py-2.5 px-3 text-center font-bold text-emerald-400">88대</td>
                                    <td class="py-2.5 px-3 text-right text-slate-300">평균 8.8 km</td>
                                    <td class="py-2.5 px-3 text-right font-mono text-cyan-300">평균 51.4 분</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 5: BAYESIAN BROKEN BIKE DETECTION -->
        <section id="bayesian" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-amber-400 uppercase tracking-widest">Section 05</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">공공 API 스냅샷 기반 베이지안 고장 자전거 감지</h2>
                <p class="text-sm text-slate-400 mt-1">자전거 고유 ID가 없는 실시간 API 환경에서, 포아송 생존 분석으로 1대 고착 유령 자전거를 99.9% 확정하는 알고리즘</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <h3 class="text-lg font-bold text-white">알고리즘 수식 및 추론 원리</h3>
                    <div class="bg-slate-950 p-4 rounded-xl font-mono text-xs text-amber-300 border border-slate-800 leading-relaxed">
                        P(고장 | t시간 고착) = 1 - exp(-λ × t)<br/>
                        • λ: 해당 대여소의 해당 요일/시간대 평균 시간당 대여 수요 (대/h)<br/>
                        • t: 재고가 1대로 유지되며 연속 정체된 시간 (h)
                    </div>
                    <p class="text-xs sm:text-sm text-slate-300 leading-relaxed">
                        공공 API에는 자전거 ID가 제공되지 않아, "대여소에 1대가 남아있는 것"이 정상 자전거를 시민들이 안 타는 것인지, 
                        <strong>체인이 빠지거나 펑크가 나서 아무도 못 타는 고장 자전거</strong>인지 알 수 없었습니다.
                    </p>
                    <p class="text-xs sm:text-sm text-slate-300 leading-relaxed">
                        저희 시스템은 시간당 대여 수요가 4~8대 이상인 인기 대여소에서 <strong>1대가 2시간 이상 정체될 경우, 정상일 확률이 0.0003(0.03%) 미만으로 떨어지는 베이지안 생존 분석</strong>을 적용합니다. 
                        사후 확률 99.9%에 도달하는 즉시 관제 UI에서 유효 재고를 0으로 차감하고 트럭 수거 명령을 생성합니다.
                    </p>
                </div>

                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">시간당 수요(λ) 및 정체 시간에 따른 고장 확정 확률</h4>
                    
                    <div class="space-y-3">
                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-slate-300 font-medium">인기 대여소 (λ = 6.0 대/h, 1시간 정체)</span>
                                <span class="font-mono text-amber-400 font-bold">99.75% 고장 확정</span>
                            </div>
                            <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                                <div class="bg-amber-400 h-full rounded-full" style="width: 99.75%"></div>
                            </div>
                        </div>

                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-slate-300 font-medium">표준 대여소 (λ = 3.0 대/h, 2시간 정체)</span>
                                <span class="font-mono text-amber-400 font-bold">99.75% 고장 확정</span>
                            </div>
                            <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                                <div class="bg-amber-400 h-full rounded-full" style="width: 99.75%"></div>
                            </div>
                        </div>

                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-slate-300 font-medium">일반 대여소 (λ = 1.5 대/h, 3시간 정체)</span>
                                <span class="font-mono text-cyan-400 font-bold">98.89% 고장 확정</span>
                            </div>
                            <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                                <div class="bg-cyan-400 h-full rounded-full" style="width: 98.89%"></div>
                            </div>
                        </div>

                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-slate-300 font-medium">외곽 저수요 대여소 (λ = 0.5 대/h, 1시간 정체)</span>
                                <span class="font-mono text-slate-400">39.35% (오탐 방지 보류)</span>
                            </div>
                            <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                                <div class="bg-slate-600 h-full rounded-full" style="width: 39.35%"></div>
                            </div>
                        </div>
                    </div>

                    <div class="p-3 bg-slate-900 rounded-xl border border-slate-800 text-[11px] text-slate-400 leading-tight">
                        💡 <strong>오탐(False Positive) 방지 메커니즘:</strong> 수요가 원래 적은 외곽 대여소는 1대가 서 있어도 고장으로 섣불리 단정하지 않아 불필요한 트럭 헛걸음을 원천 차단합니다.
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 6: URBAN MOBILITY SYNERGY & ESG -->
        <section id="synergy" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Section 06</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">도시교통 인프라 연계 & ESG 탄소 감축</h2>
                <p class="text-sm text-slate-400 mt-1">2028년 개통 예정 대전 도시철도 2호선 트램 연계, 자전거 도로 단절 해소, 그리고 공인 탄소 배출권 정량화</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <!-- Card 1: Tram Line 2 -->
                <div class="glass-panel p-6 rounded-2xl space-y-3 border-cyan-500/20">
                    <div class="text-xs text-cyan-400 font-bold uppercase tracking-wider">Future Mobility</div>
                    <h3 class="text-base font-bold text-white">대전 트램 2호선 연계</h3>
                    <div class="text-2xl font-black text-cyan-300">42.4% <span class="text-xs font-normal text-slate-400">(386만 건)</span></div>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        2028년 개통될 대전 도시철도 2호선(38.8km 순환선, 45개 역)의 반경 500m 내에 <strong>전체 타슈 통행의 42.4%</strong>가 집중되어 있습니다. 트램 개통 시 타슈는 대전의 핵심 피더(Feeder) 모빌리티로 도약합니다.
                    </p>
                </div>

                <!-- Card 2: Subway Line 1 -->
                <div class="glass-panel p-6 rounded-2xl space-y-3 border-emerald-500/20">
                    <div class="text-xs text-emerald-400 font-bold uppercase tracking-wider">Transit Feeder</div>
                    <h3 class="text-base font-bold text-white">지하철 1호선 환승 편익</h3>
                    <div class="text-2xl font-black text-emerald-300">19.4% <span class="text-xs font-normal text-slate-400">(176.6만 건)</span></div>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        22개 지하철역 반경 300m 이내에서 연간 176만 건의 퍼스트·라스트마일 환승 통행이 발생합니다. 버스 환승 정시성을 보완하는 결정적 대체재 역할을 수행하고 있습니다.
                    </p>
                </div>

                <!-- Card 3: Cycleway Missing Links -->
                <div class="glass-panel p-6 rounded-2xl space-y-3 border-amber-500/20">
                    <div class="text-xs text-amber-400 font-bold uppercase tracking-wider">Urban Safety</div>
                    <h3 class="text-base font-bold text-white">자전거 도로 단절구간 Top 10</h3>
                    <div class="text-2xl font-black text-amber-300">Top 10 <span class="text-xs font-normal text-slate-400">위험 구간 도출</span></div>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        통행량은 월 10만 건이 넘지만 전용 도로망이 끊겨 시민들이 차도로 내몰리는 위험 단절 구간을 OSM 도로망 매핑으로 추출, <strong>대전시 자전거 도로 우선 투자 리포트</strong>를 제공합니다.
                    </p>
                </div>

                <!-- Card 4: Crowdsourced Gamification -->
                <div class="glass-panel p-6 rounded-2xl space-y-3 border-indigo-500/20">
                    <div class="text-xs text-indigo-400 font-bold uppercase tracking-wider">Self-Healing</div>
                    <h3 class="text-base font-bold text-white">타슈 챌린지 (시민 재배치)</h3>
                    <div class="text-2xl font-black text-indigo-300">500P <span class="text-xs font-normal text-slate-400">온통대전 캐시백</span></div>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        잉여 거점(노은동)에서 부족 거점(유성온천역)으로 출퇴근하는 시민에게 지역화폐 마일리지를 지급하여, 트럭 투입 없이 시민이 스스로 도시를 치유하는 <strong>크라우드소싱 자가치유망</strong>을 구축합니다.
                    </p>
                </div>
            </div>
        </section>

        <!-- SECTION 7: CODE, REPRODUCIBILITY & TECH STACK -->
        <section id="code" class="space-y-8 scroll-mt-24">
            <div class="border-b border-slate-800 pb-4">
                <span class="text-xs font-bold text-slate-400 uppercase tracking-widest">Section 07</span>
                <h2 class="text-2xl sm:text-3xl font-bold text-white mt-1">프로젝트 소스코드 및 실행 가이드 (Reproducibility)</h2>
                <p class="text-sm text-slate-400 mt-1">모든 파이프라인은 GitHub Branch [성시준]에 모듈화되어 즉시 재현 가능하도록 구축되었습니다.</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Code Commands -->
                <div class="glass-panel p-6 rounded-2xl lg:col-span-2 space-y-4">
                    <h3 class="text-base font-bold text-white flex items-center justify-between">
                        <span>파이프라인 실행 명령어 (CLI Commands)</span>
                        <span class="text-xs text-slate-400 font-mono">Python 3.11</span>
                    </h3>
                    
                    <div class="space-y-3 font-mono text-xs">
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <span class="text-slate-500"># 1. 300m Super-Station 군집화 및 XGBoost Q85 최종 모델 학습</span>
                            <div class="text-cyan-400 font-bold mt-1">python scripts/build_final_model.py</div>
                        </div>

                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <span class="text-slate-500"># 2. 퇴근길(17:00) 결품 방어 10대 트럭 물리 동선 최적화 (Fleet Routing)</span>
                            <div class="text-emerald-400 font-bold mt-1">python scripts/optimize_fleet_routing.py</div>
                        </div>

                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <span class="text-slate-500"># 3. 실시간 API 스냅샷 기반 1대 고착 베이지안 고장 감지 파이프라인</span>
                            <div class="text-amber-400 font-bold mt-1">python scripts/simulate_september_api_pipeline.py</div>
                        </div>

                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <span class="text-slate-500"># 4. ESG 탄소 감축량 & 도시철도 1호선 / 2호선 트램 시너지 분석</span>
                            <div class="text-indigo-400 font-bold mt-1">python scripts/calculate_esg_and_subway_feeder.py && python scripts/analyze_tram_line2_synergy.py</div>
                        </div>
                    </div>
                </div>

                <!-- Tech Stack Badges -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <h3 class="text-base font-bold text-white">기술 스택 & 의존성</h3>
                    <div class="space-y-3 text-xs">
                        <div>
                            <span class="text-slate-400 block mb-1.5 font-medium">데이터 처리 & 공간 분석</span>
                            <div class="flex flex-wrap gap-1.5">
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Polars</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Pandas</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Scipy (Spatial)</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">OSMnx</span>
                            </div>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-1.5 font-medium">머신러닝 & 비대칭 최적화</span>
                            <div class="flex flex-wrap gap-1.5">
                                <span class="px-2.5 py-1 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono font-bold">XGBoost (Q85)</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Scikit-Learn</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Bayesian Poisson</span>
                            </div>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-1.5 font-medium">물류 동선 & 시각화</span>
                            <div class="flex flex-wrap gap-1.5">
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Google OR-Tools</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Chart.js</span>
                                <span class="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Tailwind CSS</span>
                            </div>
                        </div>
                    </div>

                    <div class="pt-4 border-t border-slate-800">
                        <a href="https://github.com/sijoon-sung/KT-X-AI-/tree/성시준" target="_blank" class="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs flex items-center justify-center gap-2 transition border border-slate-700">
                            <span>GitHub 레포지토리 바로가기</span>
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
                        </a>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <!-- FOOTER -->
    <footer class="border-t border-slate-800/80 bg-slate-950 py-12 text-center text-xs text-slate-500 space-y-3">
        <div class="flex items-center justify-center gap-2">
            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            <span class="font-bold text-slate-300">KT-X-AI 대전 타슈 지능형 모빌리티 솔루션</span>
        </div>
        <p>프로젝트 총괄: 성시준 (Sijun Sung) | 페어 프로그래밍 AI: Google Deepmind Antigravity</p>
        <p class="text-[11px] text-slate-600">본 대시보드는 9,116,462건의 공공 통행 데이터와 기상 데이터를 실시간 결합하여 렌더링된 정적 웹 포트폴리오입니다.</p>
    </footer>

    <!-- CHARTS JAVASCRIPT LOGIC -->
    <script>
        // Common Chart Defaults
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Pretendard, sans-serif';

        // 1. Distance Chart (Donut)
        const ctxDist = document.getElementById('distanceChart').getContext('2d');
        new Chart(ctxDist, {{
            type: 'doughnut',
            data: {{
                labels: ['1km 이내 (초단거리)', '1~2km (생활권 표준)', '2~4km (중거리 환승)', '4km 초과 (장거리)'],
                datasets: [{{
                    data: [51.2, 24.3, 14.1, 10.4],
                    backgroundColor: ['#06b6d4', '#10b981', '#6366f1', '#f59e0b'],
                    borderWidth: 0,
                    hoverOffset: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            boxWidth: 12,
                            padding: 14,
                            font: {{ size: 11 }}
                        }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.label + ': ' + context.parsed + '%';
                            }}
                        }}
                    }}
                }},
                cutout: '65%'
            }}
        }});

        // 2. Hourly Diurnal Rhythm Chart (Line)
        const ctxHourly = document.getElementById('hourlyChart').getContext('2d');
        const weekdayHours = {list(eda['weekday_hourly'].values())};
        const weekendHours = {list(eda['weekend_hourly'].values())};
        const hoursLabels = Array.from({{length: 24}}, (_, i) => i + '시');

        new Chart(ctxHourly, {{
            type: 'line',
            data: {{
                labels: hoursLabels,
                datasets: [
                    {{
                        label: '평일 (M자 통근 쌍봉)',
                        data: weekdayHours,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    }},
                    {{
                        label: '주말 (완만한 종형 여가)',
                        data: weekendHours,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                scales: {{
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{
                            callback: function(val) {{ return (val / 10000).toFixed(0) + '만'; }}
                        }}
                    }},
                    x: {{
                        grid: {{ display: false }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{ boxWidth: 12, font: {{ size: 11 }} }}
                    }}
                }}
            }}
        }});

        // 3. Weather Sensitivity Bar Chart
        const ctxWeather = document.getElementById('weatherChart').getContext('2d');
        new Chart(ctxWeather, {{
            type: 'bar',
            data: {{
                labels: ['비 안 옴 (0mm)', '이슬비 (0~1mm)', '보통 비 (1~3mm)', '강한 비 (3~10mm)', '폭우 (>10mm)'],
                datasets: [{{
                    label: '주간 시간당 통행량 (건/h)',
                    data: [936.3, 650.1, 272.5, 266.4, 159.8],
                    backgroundColor: [
                        '#06b6d4',
                        '#38bdf8',
                        '#f43f5e',
                        '#e11d48',
                        '#be123c'
                    ],
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        title: {{ display: true, text: '시간당 이용건수', font: {{ size: 10 }} }}
                    }},
                    x: {{
                        grid: {{ display: false }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(context) {{
                                if (context.dataIndex === 2) return '🚨 -70.9% 급락 임계점';
                                if (context.dataIndex === 1) return '-30.6% 감소';
                                if (context.dataIndex >= 3) return '-71%~-83% 급감';
                                return '기준 통행량';
                            }}
                        }}
                    }}
                }}
            }}
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
