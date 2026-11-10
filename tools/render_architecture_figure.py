"""Draw source-aligned paper figures. No data, weights, or inference are loaded.

Requires matplotlib. Figures use synthetic glyphs to denote tensors/distributions,
not measured data or model predictions. All layer sizes come from the fixed contract.
"""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
C = {'ink':'#20252c','muted':'#65717e','line':'#9aa5af',
     'blue':'#356ba0','blue_bg':'#eaf2fa','purple':'#8055a5','purple_bg':'#f3edf8',
     'gold':'#ab761d','gold_bg':'#fcf3df','teal':'#267d77','teal_bg':'#e7f5f2',
     'gray_bg':'#f4f5f7'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                     'savefig.facecolor':'white','mathtext.fontset':'dejavusans'})

def canvas(w,h):
    fig=plt.figure(figsize=(w/100,h/100),facecolor='white')
    ax=fig.add_axes((0,0,1,1));ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
    return fig,ax

def text(ax,x,y,s,size=13,color='ink',weight='normal',ha='center',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=C.get(color,color),fontweight=weight,
                   ha=ha,va=va,linespacing=1.5,**kw)

def box(ax,x,y,w,h,label,sub='',color='blue',size=14,sub_size=11):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=C.get(color+'_bg','white'),
                           edgecolor=C.get(color,C['line']),lw=1.3,zorder=3))
    text(ax,x+w/2,y+h*(.39 if sub else .5),label,size,weight='bold',zorder=4)
    if sub:text(ax,x+w/2,y+h*.73,sub,sub_size,color='muted',zorder=4)

def arrow(ax,points,color='ink',dashed=False,lw=1.35,head=True):
    col=C.get(color,color)
    if len(points)>2:
        ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color=col,lw=lw,
                linestyle=(0,(4,3)) if dashed else '-',zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if head else '-',
                 mutation_scale=12,lw=lw,color=col,linestyle=(0,(4,3)) if dashed else '-',
                 shrinkA=0,shrinkB=0,zorder=2))

def frame(ax,x,y,w,h,label,color):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='none',edgecolor=C[color],lw=1,
                           linestyle=(0,(5,4)),zorder=0))
    text(ax,x+15,y+23,label,14,color=color,weight='bold',ha='left')

def matrix(ax,x,y,w,h,rows=7,cols=12,color='blue'):
    for r in range(rows):
        for c in range(cols):
            alpha=.13+.61*((r*7+c*11+r*c*3)%19)/18
            ax.add_patch(Rectangle((x+c*w/cols,y+r*h/rows),w/cols-.8,h/rows-.8,
                                  facecolor=C[color],alpha=alpha,edgecolor='none'))
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=C[color],lw=.7))

def graph(ax,x,y,scale=1):
    nodes=[(0,30),(55,0),(96,37),(70,88),(13,94),(46,49)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(1,5),(2,5),(3,5),(4,5)]
    for a,b in edges:
        ax.plot([x+nodes[a][0]*scale,x+nodes[b][0]*scale],
                [y+nodes[a][1]*scale,y+nodes[b][1]*scale],color=C['blue'],alpha=.65,lw=1)
    for k,(a,b) in enumerate(nodes):
        ax.add_patch(Circle((x+a*scale,y+b*scale),7*scale,facecolor=C['blue'] if k==5 else '#b5cce1',
                            edgecolor='white',lw=1.1,zorder=3))

def bars(ax,x,y,w,h,color='teal'):
    vals=[.12,.29,.57,.85,1,.78,.55,.32,.15,.07]
    ax.plot([x,x+w],[y+h,y+h],color=C['muted'],lw=.8)
    for j,v in enumerate(vals):
        ax.add_patch(Rectangle((x+j*w/len(vals)+2,y+h-v*h),w/len(vals)-4,v*h,
                              facecolor=C[color],alpha=.73,lw=0))

def save(fig,stem):
    OUT.mkdir(exist_ok=True)
    for fmt in ['svg','pdf','png']:
        fig.savefig(OUT/(stem+'.'+fmt),dpi=220,metadata={'Creator':'Tashu architecture figure renderer'})
        if fmt=='svg':
            svg=OUT/(stem+'.svg')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

def overview():
    fig,ax=canvas(1800,980)
    text(ax,40,37,'GCN-conditioned count forecasting with Flow Matching',23,weight='bold',ha='left')
    text(ax,40,72,'H13  |  next 10 minutes  |  fixed inputs',12,color='muted',ha='left')
    frame(ax,35,110,615,650,'(a) Graph-conditioned encoder','blue')
    frame(ax,695,110,655,650,'(b) Count decoders','purple')
    frame(ax,1395,110,370,650,'(c) Inventory projection','teal')

    # Historical tensors and an explicit one-hop graph operation.
    matrix(ax,65,193,115,120)
    text(ax,122,167,'Weekly profiles',13,weight='bold')
    text(ax,122,339,r'$P\in\mathbb{R}^{N\times420}$',13)
    box(ax,225,201,155,103,'Weekly MLP',r'$420\to256\to64$')
    arrow(ax,[(184,252),(223,252)],'blue')
    text(ax,432,251,r'$Z$',19,color='blue')
    arrow(ax,[(382,252),(416,252)],'blue')
    graph(ax,496,201,.95)
    text(ax,543,334,r'$\bar Z=AZ$',15,color='blue')
    arrow(ax,[(450,252),(479,252)],'blue')
    text(ax,543,177,'Fixed OD graph A',12,color='muted')

    box(ax,65,432,260,110,'Existing context','priors / calendar / phase\nweather / recent net + masks',size=14)
    box(ax,385,431,225,111,'Context fusion',r'$164,72\ \to\ 256$')
    arrow(ax,[(327,487),(383,487)],'blue')
    arrow(ax,[(432,271),(432,383),(456,383),(456,429)],'blue')
    arrow(ax,[(543,350),(543,429)],'blue')
    text(ax,497,598,r'$h_i\in\mathbb{R}^{256}$',17,color='blue')
    arrow(ax,[(497,544),(497,575)],'blue')
    text(ax,65,697,'Weekly MLP + fixed OD aggregation + context MLP',11,color='muted',ha='left')
    text(ax,65,725,'Full encoder layers are expanded in Fig. 2(a).',11,color='muted',ha='left')

    # Rate head; conditional Poisson comparator is drawn separately from FM.
    arrow(ax,[(552,598),(672,598),(672,245),(726,245)],'blue')
    arrow(ax,[(672,598),(726,598)],'blue',dashed=True)
    box(ax,728,199,181,92,'Rate head',r'$256\to2$','gold')
    arrow(ax,[(911,245),(940,245)],'gold')
    text(ax,987,233,r'$\lambda_i$',22,color='gold')
    text(ax,987,271,'rental / return',10,color='muted')
    box(ax,1081,192,235,100,'Poisson reference',r'$Y_R,Y_D\sim\mathrm{Pois}(\lambda_i)$','gold',13)
    arrow(ax,[(1011,245),(1079,245)],'gold')
    text(ax,988,323,r'$\lambda=p\,\exp(2\tanh(a/2))$',12,color='gold')
    text(ax,1198,332,'Separate sampling route',11,color='muted')
    text(ax,1019,167,'Poisson likelihood during count training',10,color='muted')

    # FM's conditional input and Gaussian source are unambiguously separate.
    ax.add_patch(Rectangle((726,405),596,282,fill=False,edgecolor=C['purple'],lw=1.1))
    text(ax,744,430,'Conditional Flow Matching',15,color='purple',weight='bold',ha='left')
    box(ax,915,482,227,104,r'$v_\theta(r_s,s\mid h_i,\mu_i)$','264 → 128 → 128 → 128 → 2','purple',14,sub_size=9.8)
    arrow(ax,[(987,345),(1174,345),(1174,462),(1088,462),(1088,480)],'gold',dashed=True)
    text(ax,1070,374,r'$\mu_i=\log(1+\lambda_i)$',11,color='gold')
    arrow(ax,[(728,598),(880,598),(880,462),(953,462),(953,480)],'blue',dashed=True)
    text(ax,782,619,r'$h_i$',14,color='blue')
    # Schematic Gaussian cloud: glyph positions, never model samples.
    for k in range(32):
        angle=k*2.39996;rad=math.sqrt((k+.5)/32)
        ax.add_patch(Circle((806+math.cos(angle)*rad*31,533+math.sin(angle)*rad*30),1.9,
                            facecolor=C['purple'],alpha=.6))
    text(ax,806,478,'Gaussian noise',11)
    text(ax,806,575,r'$r_0\sim\mathcal{N}(0,I_2)$',12)
    arrow(ax,[(848,534),(913,534)],'purple')
    arrow(ax,[(1144,534),(1203,534)],'purple')
    text(ax,1249,525,r'$r_1$',21,color='purple')
    text(ax,1249,565,'log residual',10,color='muted')
    text(ax,1028,619,'ODE sampling: 16 Heun steps',12,color='purple')
    text(ax,1028,652,'128 IID particles across seeds 11 and 29',11,color='muted')
    ax.add_patch(Rectangle((726,696),596,42,facecolor='white',edgecolor=C['purple'],lw=1,zorder=3))
    text(ax,746,717,r'$\widehat{Y}=\max(0,\lfloor\exp(r_1+\mu_i)-1\rfloor)$',13,color='purple',ha='left',zorder=4)
    arrow(ax,[(1250,584),(1250,694)],'purple')
    text(ax,1317,750,'128 × 2 counts',10,color='muted',ha='right')

    # Current observed stock initializes a deterministic balance projection.
    box(ax,1430,209,300,92,'Current inventory',r'$b_{i,t}$  (API observation)','gray',14)
    box(ax,1430,430,300,113,'Inventory balance',r'$b_{i,t+10}=\max(0,b_{i,t}+Y_D-Y_R)$','teal',14)
    arrow(ax,[(1580,303),(1580,428)],'teal')
    arrow(ax,[(1580,545),(1580,568)],'teal')
    arrow(ax,[(1324,717),(1370,717),(1370,487),(1428,487)],'purple')
    text(ax,1375,577,'counts',10,color='purple',ha='left',rotation=90)
    bars(ax,1480,582,200,77)
    text(ax,1580,683,'Median / interval / P(stock = 0)',11,color='teal')
    text(ax,1580,723,'Inventory distribution (128 particles)',10,color='muted')
    arrow(ax,[(1580,763),(1580,804)],'teal')
    box(ax,1405,806,350,72,'Dispatch optimizer','pickup / delivery quantities and route','gray',13)

    # Training-only strip is separate from forward inference.
    text(ax,40,804,'TRAINING',10,color='muted',weight='bold',ha='left')
    box(ax,40,833,295,66,'Seoul count pretraining','initialize weekly encoder','blue',12)
    box(ax,395,833,330,66,'Tashu historical count training','fit encoder + rate head / Poisson NLL','gold',12)
    box(ax,785,833,330,66,'Freeze count network; train FM','residual velocity matching / MSE','purple',12)
    arrow(ax,[(337,866),(393,866)],'muted');arrow(ax,[(727,866),(783,866)],'muted')
    text(ax,40,946,'N: stations. R: rentals. D: returns. Dashed arrows: conditioning. Distribution glyphs are schematic.',10,color='muted',ha='left')
    save(fig,'model-architecture-v1')

def details():
    fig,ax=canvas(1800,1080)
    text(ax,40,36,'H13 encoder and Flow Matching decoder',23,weight='bold',ha='left')
    text(ax,40,73,'Exact layer dimensions in the current implementation',12,color='muted',ha='left')
    frame(ax,35,111,1730,463,'(a) Encoder: weekly profile, fixed graph aggregation, and parallel projections','blue')
    box(ax,65,181,158,88,'Weekly profile',r'$P:N\times420$','blue',13)
    box(ax,269,181,200,88,'Weekly encoder','Linear 420 → 256 → 64\nSiLU / LayerNorm','blue',13,sub_size=10)
    text(ax,550,225,r'$Z:N\times64$',15,color='blue')
    arrow(ax,[(225,225),(267,225)],'blue');arrow(ax,[(471,225),(493,225)],'blue')
    box(ax,630,181,202,88,'OD aggregation',r'$\bar Z=AZ:N\times64$','blue',13)
    arrow(ax,[(608,225),(628,225)],'blue')
    text(ax,100,316,'Own input',12,weight='bold',ha='left')
    text(ax,100,353,'prior6 + calendar35 + phase3 + weather54 + z64 + recent2',12,ha='left')
    text(ax,100,419,'Neighbor input',12,weight='bold',ha='left')
    text(ax,100,456,'prior6 + (Az)64 + recent2',12,ha='left')
    arrow(ax,[(550,271),(550,295),(701,295),(701,315)],'blue',dashed=True)
    arrow(ax,[(732,271),(868,271),(868,411),(903,411)],'blue',dashed=True)
    box(ax,905,313,200,79,r'$W_o x_i+b_o$','Linear 164 → 256','blue',14)
    box(ax,905,416,200,79,r'$W_n\bar x_i$','Linear 72 → 256 / no bias','blue',14,sub_size=10)
    arrow(ax,[(748,353),(903,353)],'blue');arrow(ax,[(674,456),(903,456)],'blue')
    ax.add_patch(Circle((1175,405),19,facecolor='white',edgecolor=C['blue'],lw=1.4,zorder=3))
    text(ax,1175,404,'+',22,color='blue')
    arrow(ax,[(1107,353),(1175,353),(1175,383)],'blue')
    arrow(ax,[(1107,456),(1175,456),(1175,426)],'blue')
    box(ax,1243,365,134,79,'SiLU','','blue',14)
    box(ax,1420,348,195,113,'Context MLP','[Linear 256 → 256\n+ SiLU] × 2','blue',13)
    arrow(ax,[(1196,405),(1241,405)],'blue');arrow(ax,[(1379,405),(1418,405)],'blue')
    arrow(ax,[(1617,405),(1662,405)],'blue')
    text(ax,1699,405,r'$h_i$',20,color='blue')
    text(ax,1699,443,'256',11,color='muted')
    text(ax,70,538,'The graph operator A is fixed from historical OD records. Calendar, weather and recent-value encodings are unchanged.',11,color='muted',ha='left')

    frame(ax,35,609,1730,305,'(b) Decoder: residual velocity network','purple')
    box(ax,65,686,318,126,'Concatenate',r'$r_s^2,\ \tau(s)^4,\ h_i^{256},\ \mu_i^2$'+'\n264 dimensions','purple',14)
    xs=[440,706,972]
    for x in xs:box(ax,x,695,211,108,'Linear + SiLU','264 → 128' if x==440 else '128 → 128','purple',13)
    box(ax,1238,695,184,108,'Linear','128 → 2','purple',14)
    box(ax,1480,695,249,108,r'$v_\theta\in\mathbb{R}^2$','residual velocity','purple',16)
    for l,r in [(385,438),(653,704),(919,970),(1185,1236),(1424,1478)]:arrow(ax,[(l,749),(r,749)],'purple')
    text(ax,70,858,r'$\mu_i=\log(1+\lambda_i),\qquad\tau(s)=[s,\sin(\pi s),\cos(\pi s),s^2]$',13,ha='left')
    text(ax,70,960,'FM training',12,color='purple',weight='bold',ha='left')
    text(ax,245,960,r'$r_1=\log(1+y+u)-\mu_i,\quad r_s=(1-s)r_0+sr_1,\quad \mathcal{L}_{\rm FM}=\|v_\theta(r_s,s\mid h_i,\mu_i)-(r_1-r_0)\|^2$',14,ha='left')
    text(ax,70,1000,r'$r_0\sim\mathcal{N}(0,I_2),\quad u\sim U([0,1)^2),\quad s\sim U([0,1])$.  The count network is frozen while training FM.',12,color='muted',ha='left')
    text(ax,70,1043,'Reference implementation: forecast_transfer.py / rolling_inventory_model.py / scaled_flow_core.py (5db8c21).',10,color='muted',ha='left')
    save(fig,'model-architecture-layers-v1')

if __name__=='__main__':
    overview();details()
    print('Rendered two architecture figures as SVG, PDF and PNG. No model execution.')
