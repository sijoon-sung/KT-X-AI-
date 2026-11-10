"""Plain-language diagram of the locked H13 model. Render on dislab."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(18,10),dpi=150);fig.patch.set_facecolor('#FAFCFC');ax.set(xlim=(0,1800),ylim=(0,1000));ax.axis('off');fig.subplots_adjust(0,0,1,1)
    dark='#18353C';muted='#577078';teal='#087F83'
    def label(x,y,text,size=20,color=dark,weight='normal',ha='left'):
        ax.text(x,y,text,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',linespacing=1.5)
    def box(x,y,w,h,fill,title,body,tag):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=16',facecolor=fill,edgecolor='#BCD0D3',linewidth=1.1))
        label(x+22,y+h-36,tag,15,muted)
        label(x+22,y+h-78,title,23,weight='bold')
        label(x+22,y+58,body,17)
    def arrow(x1,y1,x2,y2,color=teal,style='-'):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=19,linewidth=2,color=color,linestyle=style,connectionstyle='arc3'))
    label(62,944,'타슈는 10분 뒤를 어떻게 예상할까?',34,weight='bold')
    label(64,891,'주변 흐름을 읽고, 빌리고 돌려줄 대수를 예상해 여러 미래를 만듭니다.',21,muted)
    box(60,510,300,275,'#FFFFFF','이용 기록과 주변 상황','지난 대여·반납 기록\n날씨·요일·시간','입력 자료')
    box(420,510,370,275,'#EAF4FD','주변 대여소도 함께 보기','연결된 대여소의 흐름을\n한 대여소의 상황과 함께 정리','1  GCN · 상황 정리(encoder)')
    box(850,510,370,275,'#FFF3E4','빌리고 돌려줄 대수 예상','대여·반납이 각각\n평균적으로 얼마나 생길까?','2  포아송 평균')
    box(1280,510,460,275,'#E7F4EF','가능한 미래를 여러 개 만들기','주변 상황과 예상 평균을 보고\n대여·반납의 여러 가능성을 생성','3  Flow Matching · 미래 생성(decoder)')
    for x1,x2 in ((360,420),(790,850),(1220,1280)):arrow(x1,647,x2,647)
    ax.plot([605,605,1510,1510],[785,827,827,785],color=teal,linewidth=1.5)
    arrow(1510,807,1510,785)
    label(1058,850,'GCN이 정리한 주변 상황도 FM에 전달',16,muted,ha='center')
    label(1510,480,'FM은 Gaussian 잡음에서 시작합니다.',14,muted,ha='center')
    box(60,235,300,185,'#FFFFFF','지금 남은 자전거','가장 최근에 받은 재고','현재 상태 · 재고 API')
    box(850,235,890,185,'#FFFFFF','10분 뒤 남을 자전거와, 자전거가 없을 가능성','지금 대수 + 예상 반납 - 예상 대여  →  여러 재고 표본','예측 결과')
    arrow(360,326,850,326);arrow(1510,510,1510,420)
    label(600,354,'현재 재고는 마지막 계산에 반영',16,muted,ha='center')
    label(60,169,'이 결과를 서비스에서 어떻게 쓸까?',22,weight='bold')
    label(65,110,'부족할 곳 찾기',22,weight='bold');arrow(350,110,420,110)
    label(450,110,'이동 대수·방문 순서 계산',22,weight='bold');arrow(955,110,1025,110)
    label(1050,110,'외부 LLM이 쉬운 말로 설명',22,weight='bold')
    label(65,45,'예측 모델은 고정합니다. 새 재고가 오면 다시 예측합니다. LLM은 예측값을 바꾸지 않습니다.',17,muted)
    for ext in ('svg','png','pdf'):fig.savefig(out/f'tashu-simple-architecture.{ext}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
