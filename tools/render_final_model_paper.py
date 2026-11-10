"""Paper figures for the retained H13 backbone. Render on dislab only.

Glyphs are schematic; no data, weights, fitted curves or accuracy are loaded.
Visual grammar: Graph WaveNet Fig3, Kipf/Welling Fig1; FM equations: Lipman et al.
"""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,Circle,Ellipse,FancyArrowPatch,PathPatch
from matplotlib.path import Path as Curve

P={'ink':'#20252B','muted':'#66717E','line':'#8994A0','blue':'#456F99','blue_bg':'#EBF1F7',
   'purple':'#80629D','purple_bg':'#F1EDF6','gold':'#A58032','gold_bg':'#FAF4E5',
   'teal':'#377C78','teal_bg':'#EBF4F1','gray':'#EEF0F2','red':'#A85349'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.1,'mathtext.fontset':'stix',
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65})

def canvas(height):
    f=plt.figure(figsize=(7.4,7.4*height/1000),facecolor='white')
    a=f.add_axes([0,0,1,1]);a.set(xlim=(0,1000),ylim=(height,0));a.axis('off')
    return f,a

def t(a,x,y,s,size=7.1,color='ink',ha='center',weight='normal',va='center',**kw):
    return a.text(x,y,s,fontsize=size,color=P.get(color,color),ha=ha,va=va,fontweight=weight,linespacing=1.3,**kw)

def rect(a,x,y,w,h,label='',sub='',color='blue',size=7.0,fill=True):
    a.add_patch(Rectangle((x,y),w,h,facecolor=P.get(color+'_bg','white') if fill else 'white',edgecolor=P.get(color,color),lw=.7,zorder=3))
    if label:
        label_text=t(a,x+w/2,y+h*(.36 if sub else .5),label,size,zorder=4)
        label_text._paper_max_width=w*.92
    if sub:
        sub_text=t(a,x+w/2,y+h*.76,sub,6.25,color='muted',zorder=4)
        sub_text._paper_max_width=w*.92

def arr(a,points,color='ink',dash=False,lw=.72,head=True):
    col=P.get(color,color);style=(0,(3,2)) if dash else '-'
    if len(points)>2:a.plot(*zip(*points[:-1]),color=col,lw=lw,linestyle=style,zorder=1)
    a.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if head else '-',mutation_scale=6,lw=lw,color=col,linestyle=style,shrinkA=0,shrinkB=0,zorder=2))

def dot(a,x,y,r=3,color='blue'):
    a.add_patch(Circle((x,y),r,facecolor=P.get(color,color),edgecolor='white',lw=.4,zorder=5))

def tensor(a,x,y,w,h,rows=5,cols=7,color='blue'):
    for k in (2,1,0):
        dx=k*5;dy=-k*4
        a.add_patch(Rectangle((x+dx,y+dy),w,h,facecolor='white',edgecolor=P[color],lw=.65,zorder=2))
    for r in range(rows):
        for c in range(cols):
            alpha=.12+.53*((r*7+c*5+r*c)%13)/12
            a.add_patch(Rectangle((x+c*w/cols+.8,y+r*h/rows+.8),w/cols-1.6,h/rows-1.6,facecolor=P[color],alpha=alpha,edgecolor='none',zorder=3))

def graph(a,x,y,scale=.7):
    xy=[(0,25),(37,0),(77,18),(88,62),(44,86),(2,70),(42,40)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,6),(1,6),(2,6),(3,6),(4,6),(5,6)]
    for i,j in edges:a.plot([x+xy[i][0]*scale,x+xy[j][0]*scale],[y+xy[i][1]*scale,y+xy[j][1]*scale],color=P['blue'],alpha=.50,lw=.65,zorder=1)
    for i,(u,v) in enumerate(xy):dot(a,x+u*scale,y+v*scale,4 if i==6 else 3.4,'blue' if i==6 else '#A8BDCF')

def hist(a,x,y,w,h,color='teal'):
    vals=[.82,.95,.64,.36,.18,.08,.025]
    a.plot([x,x+w],[y+h,y+h],color=P['line'],lw=.55)
    for j,v in enumerate(vals):a.add_patch(Rectangle((x+j*w/7+1,y+h-h*v),w/7-2,h*v,facecolor=P[color],alpha=.70,edgecolor='none'))

def plus(a,x,y):
    a.add_patch(Circle((x,y),10,facecolor='white',edgecolor=P['ink'],lw=.7,zorder=3))
    a.plot([x-4,x+4],[y,y],color=P['ink'],lw=.7,zorder=4);a.plot([x,x],[y-4,y+4],color=P['ink'],lw=.7,zorder=4)

def panel(a,x,y,label,title):
    t(a,x,y,label,8.3,weight='bold',ha='left');t(a,x+33,y,title,7.7,weight='bold',ha='left')

def source_cloud(a,x,y,color='purple'):
    a.add_patch(Ellipse((x,y),52,67,facecolor=P[color+'_bg'],edgecolor=P[color],lw=.65))
    for j in range(19):
        angle=j*2.39996;radius=6+12*((j*7)%13)/13
        dot(a,x+radius*math.cos(angle),y+1.3*radius*math.sin(angle),1.45,color)

def save(f,a,out,stem):
    f.canvas.draw();renderer=f.canvas.get_renderer();bad=[]
    too_small=[]
    for text in a.texts:
        limit=getattr(text,'_paper_max_width',None)
        if limit:
            bb=text.get_window_extent(renderer);allowed=limit*f.bbox.width/1000
            if bb.width>allowed:
                new_size=text.get_fontsize()*allowed/bb.width
                if new_size<5.7:too_small.append(text.get_text())
                text.set_fontsize(new_size)
    if too_small:raise ValueError('Wrap labels instead of shrinking: '+repr(too_small))
    f.canvas.draw();renderer=f.canvas.get_renderer()
    for text in a.texts:
        bb=text.get_window_extent(renderer);fb=f.bbox
        if bb.x0<fb.x0-1 or bb.y0<fb.y0-1 or bb.x1>fb.x1+1 or bb.y1>fb.y1+1:bad.append(text.get_text())
    if bad:raise ValueError('Out-of-page labels: '+repr(bad))
    for fmt in ('svg','pdf','png'):
        p=out/f'{stem}.{fmt}';f.savefig(p,dpi=600,metadata={'Creator':'Tashu source-aligned paper figure'})
        if fmt=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(f)

def overview(out):
    f,a=canvas(760)
    panel(a,18,24,'(a)','Graph-conditioned encoder')
    panel(a,372,24,'(b)','Poisson-conditioned Flow Matching')
    panel(a,775,24,'(c)','Inventory / shortfall')
    for x in (350,758):a.plot([x,x],[52,610],color='#D8DDE2',lw=.6)

    # Static station representation and a single fixed OD aggregation.
    tensor(a,30,105,65,47);t(a,62,78,'Weekly profiles',6.9)
    t(a,65,174,r'$P\in\mathbb{R}^{N\times420}$',8.0)
    arr(a,[(106,130),(150,130)])
    rect(a,150,108,110,43,'Profile MLP','420 → 256 → 64')
    arr(a,[(261,130),(292,130)]);tensor(a,295,114,25,37,rows=5,cols=2)
    t(a,310,177,r'$z_i\in\mathbb{R}^{64}$',8.0)
    graph(a,43,235,.78);t(a,80,325,'Historical OD graph',6.6)
    arr(a,[(119,270),(164,270)])
    rect(a,164,247,132,48,r'$(AZ)_i$','Fixed OD')
    arr(a,[(307,183),(307,219),(233,219),(233,247)],color='blue')
    # Current covariates enter the same own/neighbor fusion, not the graph topology.
    tensor(a,29,403,66,57,rows=6,cols=6,color='teal')
    t(a,73,492,'Prior · calendar\nWeather\nRecent proxy',6.2)
    rect(a,161,390,151,76,'Own / neighbor fusion','164 / 72 → 256',size=7.0)
    arr(a,[(102,434),(161,434)])
    arr(a,[(233,295),(233,390)],color='blue')
    arr(a,[(323,150),(336,150),(336,375),(278,375),(278,390)],color='blue')
    t(a,302,354,'self',6.0,color='muted')
    arr(a,[(237,466),(237,501)])
    rect(a,181,502,112,42,'MLP ×2','256 → 256 · SiLU')
    arr(a,[(293,523),(338,523)],color='blue');t(a,310,559,r'$h_i\in\mathbb{R}^{256}$',7.8,color='blue')

    # Explicit shared-context bus: lambda and velocity are conditioned on h.
    arr(a,[(338,523),(365,523),(365,166),(403,166)],color='blue',dash=True)
    arr(a,[(365,306),(486,306)],color='blue',dash=True)
    rect(a,403,143,155,46,'Poisson rate head',r'$\lambda_i=\lambda_i^0\,\exp(2\tanh(a_i/2))$',color='gold',size=7.1)
    arr(a,[(558,166),(591,166)],color='gold');t(a,620,166,r'$\lambda_i\in\mathbb{R}_+^2$',8.4,color='gold')
    arr(a,[(622,186),(622,231),(581,231),(581,274)],color='gold',dash=True)
    t(a,649,222,r'$c_i=\log(1+\lambda_i)$',7.1,color='gold')
    # Poisson is a parallel reference, NOT the initial FM state.
    arr(a,[(622,149),(622,99),(656,99)],color='line',dash=True)
    rect(a,656,78,81,42,'Poisson','reference',color='line',size=6.8)
    t(a,697,142,'Count baseline only',6.1,color='muted')
    rect(a,486,274,153,46,r'$v_\theta(r_s,s,h_i,c_i)$','Velocity MLP',color='purple',size=8.3)
    arr(a,[(561,320),(561,350)],color='purple',dash=True)

    # An original schematic latent path: Gaussian noise to a count residual.
    source_cloud(a,409,401);t(a,410,349,r'$r_0\sim\mathcal{N}(0,I_2)$',7.6)
    for j,col in enumerate(('#AC95C2','#80629D','#5F4680')):
        y0=385+j*15;y1=399+(j-1)*12
        verts=[(440,y0),(492,y0-35+j*12),(570,y1+45-j*20),(633,y1)]
        a.add_patch(PathPatch(Curve(verts,[Curve.MOVETO,Curve.CURVE4,Curve.CURVE4,Curve.CURVE4]),fill=False,edgecolor=col,lw=.9))
        arr(a,[(605,y1+3),(633,y1)],color=col,lw=.8)
    source_cloud(a,664,401);t(a,664,350,r'$r_1\in\mathbb{R}^{2}$',8)
    t(a,536,468,'Heun integration · 16 steps',6.6,color='purple')
    t(a,535,490,r'Generative time $s:0\rightarrow1$',6.6,color='muted')
    arr(a,[(664,439),(664,514)])
    rect(a,603,515,132,52,'Integer count\ntransform',r'$[\lfloor\exp(r_1+c_i)-1\rfloor]_+$',color='purple',size=6.7)
    t(a,666,590,r'$(\widehat R_i,\widehat D_i)^{(m)}$',8.4,color='purple')
    t(a,541,561,'128 IID particles\n2-seed mixture',6.4,color='muted')

    # Current inventory is a state input; no API labels enter historical fitting.
    tensor(a,835,132,95,43,rows=1,cols=6,color='teal');t(a,884,100,'Observed stock vector',6.8)
    t(a,884,195,r'$b_{i,t}$',9.0,color='teal')
    arr(a,[(885,211),(885,496)],color='teal')
    t(a,905,298,'+10 min',6.8,color='teal',ha='left')
    arr(a,[(735,541),(789,541)])
    t(a,770,509,'counts',6.2,color='muted')
    rect(a,789,497,192,70,'Inventory projection',r'$\hat b_{i,t+10}^{(m)}=[b_{i,t}+\hat D_i^{(m)}-\hat R_i^{(m)}]_+$',color='teal',size=7.0)
    arr(a,[(885,568),(885,591)],color='teal');hist(a,833,594,105,40,color='teal')
    t(a,885,653,'Inventory distribution',6.7,color='teal')
    arr(a,[(885,662),(885,668)],color='teal')
    rect(a,782,668,207,65,'Cost-aware shortfall readout',r'$Z=[b^*-\hat b]_+$'+'\n'+r'$q_\tau(Z),\quad \tau=c_u/(c_u+c_o)$',color='teal',size=6.6)

    # Compact two-stage learning annotation, separate from inference.
    a.plot([18,737],[625,625],color='#D8DDE2',lw=.6)
    t(a,20,645,'Historical training only',6.7,ha='left',weight='bold')
    rect(a,20,677,258,42,'1  Fit count-rate network',r'$\mathcal{L}_{\rm Pois}=\mathbb{E}[\lambda-y\log\lambda]$',color='gold',size=6.6)
    arr(a,[(280,698),(328,698)],color='muted')
    t(a,305,670,'freeze',6.1,color='muted')
    rect(a,330,677,407,42,'2  Fit FM with the rate encoder frozen',r'$\mathcal{L}_{\rm CFM}=\mathbb{E}\|v_\theta(r_s,s,h,c)-(r_1-r_0)\|^2$',color='purple',size=6.6)
    t(a,20,747,'Solid: computation   Dashed: conditioning / reference   Glyphs: schematic   Shortfall readout: evaluated postprocessing',6.0,ha='left',color='muted')
    save(f,a,out,'tashu-final-framework')

def layer_bar(a,x,y,w,label,sub='',color='blue'):
    # Small depth offsets distinguish feature tensors from scalar process boxes.
    for k in (2,1):a.add_patch(Rectangle((x+k*3,y-k*3),w,32,facecolor='white',edgecolor=P[color],lw=.5,zorder=2))
    rect(a,x,y,w,32,label,sub,color,size=7.0)

def details(out):
    f,a=canvas(800)
    panel(a,20,25,'(a)','Encoder: graph and context fusion')
    panel(a,524,25,'(b)','Decoder: conditional velocity network')
    a.plot([498,498],[58,620],color='#D8DDE2',lw=.6)
    tensor(a,27,90,64,41);t(a,60,158,r'$P:N\times420$',7.4)
    arr(a,[(101,110),(132,110)])
    rect(a,132,87,215,67,'Weekly profile encoder','420→256 · SiLU\n256→64 · LayerNorm',size=6.9)
    arr(a,[(348,110),(380,110)]);tensor(a,385,97,44,31,rows=4,cols=4)
    t(a,408,157,r'$Z:N\times64$',7.6)
    graph(a,48,191,.55);t(a,77,260,r'Fixed OD $A$',6.5)
    arr(a,[(109,220),(152,220)])
    rect(a,153,198,189,42,r'$\bar Z=A Z$','Fixed graph aggregation',size=8.0)
    arr(a,[(407,169),(407,183),(267,183),(267,198)],color='blue')
    t(a,117,294,'Own-station features',7.0,weight='bold')
    t(a,366,294,'Neighbor features',7.0,weight='bold')
    layer_bar(a,29,318,178,r'$U_i\in\mathbb{R}^{164}$',color='blue')
    layer_bar(a,275,318,178,r'$V_i\in\mathbb{R}^{72}$',color='blue')
    t(a,117,376,'6 prior · 35 calendar · 3 phase\n54 weather · 64 embedding\n2 recent',5.9,color='muted')
    t(a,366,369,'6 neighbor prior + 64 graph\nembedding + 2 recent',6.1,color='muted')
    arr(a,[(117,401),(117,411)]);arr(a,[(366,389),(366,411)])
    rect(a,36,412,162,37,r'$W_{\rm own}:164\rightarrow256$',size=7.8)
    rect(a,286,412,162,37,r'$W_{\rm nbr}:72\rightarrow256$',size=7.8)
    t(a,413,470,'no bias',6.1,color='muted')
    arr(a,[(117,449),(117,485),(230,485)]);arr(a,[(367,449),(367,485),(250,485)])
    plus(a,240,485);arr(a,[(240,495),(240,521)])
    rect(a,177,522,127,29,'SiLU',size=7.2)
    arr(a,[(240,551),(240,573)])
    layer_bar(a,125,574,230,'Linear 256→256 + SiLU, twice',color='blue')
    arr(a,[(356,590),(465,590)],color='blue');t(a,415,567,r'$h_i:256$',8,color='blue')

    # Exact concat order of scaled_flow_core.CountFlow.forward.
    inputs=[(530,65,r'$r_s$',2,'purple'),(622,78,r'$e(s)$',4,'purple'),(733,117,r'$h_i$',256,'blue'),(881,91,r'$c_i$',2,'gold')]
    for x,w,label,dim,color in inputs:
        layer_bar(a,x,112,w,label,color=color);t(a,x+w/2,166,str(dim),7.3,color=color)
        arr(a,[(x+w/2,178),(x+w/2,207),(750,207)],color=color,head=False)
    rect(a,630,214,240,34,'Concatenate in this order: 264',color='purple',size=6.9)
    arr(a,[(750,208),(750,214)],color='purple')
    t(a,750,77,r'$e(s)=[s,\sin(\pi s),\cos(\pi s),s^2]$',7.1,color='muted')
    layer_bar(a,630,316,240,'Linear 264 → 128',color='purple')
    for y,label in ((390,'Linear 128 → 128'),(464,'Linear 128 → 128'),(538,'Linear 128 → 2')):
        arr(a,[(750,y-42),(750,y-3)],color='purple');t(a,882,y-21,'SiLU',6.8,color='purple')
        layer_bar(a,630,y,240,label,color='purple')
    arr(a,[(750,248),(750,313)],color='purple')
    arr(a,[(750,570),(750,599)],color='purple')
    t(a,750,614,r'$v_\theta\in\mathbb{R}^{2}$',9.2,color='purple')
    t(a,751,647,'Shared velocity network at every Heun step',6.4,color='muted')

    a.plot([20,980],[672,672],color='#D8DDE2',lw=.6)
    panel(a,20,694,'(c)','Two-stage historical learning; fixed-weight API evaluation')
    t(a,28,727,r'$r_0\sim\mathcal{N}(0,I_2),\qquad r_1=\log(1+y+U)-\log(1+\lambda),\qquad U\sim\mathcal{U}(0,1)$',8.0,ha='left')
    t(a,28,757,r'$r_s=(1-s)r_0+s r_1,\qquad \mathcal{L}_{\rm CFM}=\mathbb{E}\|v_\theta(r_s,s,h,c)-(r_1-r_0)\|^2$',8.0,ha='left')
    t(a,28,786,'Rate network fitted first and frozen for FM. API observations enter evaluation / starting state; no gradient returns to either network.',6.35,ha='left',color='muted')
    save(f,a,out,'tashu-final-layers')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    overview(out);details(out)
    (out/'figure-render-check.json').write_text(json.dumps({'renderer':'matplotlib vector','execution_required':'dislab',
        'source':'H13 retained implementation','formats':['pdf','svg','png'],'png_dpi':600,'physical_width_inches':7.4,
        'page_boundary_checks':'passed','synthetic_glyphs_only':True,'new_data_or_model_inference_in_renderer':False},indent=2)+'\n',encoding='utf-8')
    print('FINAL_MODEL_FIGURES_RENDERED')
