"""Compose a two-minute, geographically anchored replay from private visual evidence."""
import argparse
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1920, 1080, 10
MX, MY, MW, MH = 56, 219, 1228, 694
PX, PW = 1308, 556
BG, PANEL, EDGE = '#0D1827', '#14243A', '#263D54'
WHITE, MUTED, MINT, AMBER, RED, BLUE = '#F1F6FD', '#AABED2', '#61E6CE', '#FFC479', '#FF786E', '#87BDFD'


@lru_cache(maxsize=32)
def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('malgunbd.ttf' if bold else 'malgun.ttf'),size)


def text(draw, xy, value, size, color=WHITE, bold=False, anchor=None):
    draw.text(xy,str(value),font=font(size,bold),fill=color,anchor=anchor)


def lines(draw, x, y, content, size=27, color=MUTED, gap=12):
    for row in content:
        text(draw,(x,y),row,size,color)
        y+=size+gap


def clock(value, seconds=True):
    stamp = datetime.fromisoformat(value) if isinstance(value,str) else value
    if seconds:stamp += timedelta(microseconds=500000)
    return stamp.strftime('%H:%M:%S' if seconds else '%H:%M')


def bubble(draw, x, y, value, color, radius=14):
    draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=BG,outline=color,width=3)
    if value is not None:text(draw,(x,y-1),value,int(radius*1.13),color,True,'mm')


def label(draw, x, y, value, color=WHITE):
    width=draw.textlength(value,font=font(22,True))+22
    x=max(10,min(x,MW-width-9));y=max(10,min(y,MH-54))
    draw.rounded_rectangle((x,y,x+width,y+37),radius=8,fill='#102134',outline=EDGE)
    text(draw,(x+11,y+4),value,22,color,True)


def lerp(a,b,t):return a*(1-t)+b*t


class LocalMap:
    def __init__(self,bounds,data):
        left,bottom,right,top=bounds
        self.center=((left+right)/2,(bottom+top)/2)
        self.cos=math.cos(math.radians(self.center[1]))
        self.scale=min(MW/((right-left)*self.cos),MH/(top-bottom))
        self.base=Image.new('RGB',(MW,MH),'#142236')
        draw=ImageDraw.Draw(self.base)
        elements=data['elements']
        nodes={item['id']:self.point(item['lon'],item['lat']) for item in elements if item['type']=='node'}
        roads=[];places=[]
        for item in elements:
            tags=item.get('tags',{})
            if item['type']=='node' and tags.get('railway')=='station' and tags.get('name'):
                places.append((nodes[item['id']],tags['name']))
            if item['type']!='way':continue
            points=[nodes[node] for node in item.get('nodes',[]) if node in nodes]
            if len(points)<2:continue
            closed=item['nodes'][0]==item['nodes'][-1]
            if closed and (tags.get('natural')=='water' or tags.get('waterway')=='riverbank'):
                draw.polygon(points,fill='#174052')
            elif closed and (tags.get('leisure')=='park' or tags.get('landuse') in ('forest','grass','recreation_ground')):
                draw.polygon(points,fill='#193337')
            elif closed and tags.get('building'):
                draw.polygon(points,fill='#1A2A3E')
            elif tags.get('waterway') in ('river','stream','canal'):
                roads.append((0,points,'#22556A',7,''))
            elif tags.get('railway') in ('rail','light_rail'):
                roads.append((3,points,'#5B6477',4,''))
            elif 'highway' in tags and tags['highway'] not in ('construction','proposed'):
                priority={'trunk':5,'primary':5,'secondary':4,'tertiary':3,'residential':2,'service':1}.get(tags['highway'],1)
                shades={1:('#25354A',1),2:('#34475D',2),3:('#455B71',3),4:('#54677B',5),5:('#617589',7)}
                color,width=shades[priority]
                roads.append((priority,points,color,width,tags.get('name','') if priority>=4 else ''))
        if not roads:raise ValueError('The map has no geographic road geometry')
        for _,points,color,width,_ in sorted(roads,key=lambda row:row[0]):
            draw.line(points,fill=color,width=width,joint='curve')
        names=set();positions=[]
        for priority,points,_,_,name in sorted(roads,key=lambda row:(row[0],len(row[1])),reverse=True):
            if not name or name in names or len(names)>=6:continue
            x,y=points[len(points)//2]
            if not 60<x<MW-100 or not 60<y<MH-55 or any(math.hypot(x-a,y-b)<135 for a,b in positions):continue
            text(draw,(x,y),name,18,'#879AB0',False,'mm')
            names.add(name);positions.append((x,y))
        for (x,y),name in places:
            if 10<x<MW-100 and 10<y<MH-45:text(draw,(x+8,y+8),name,21,'#B5C8DD',True)

    def point(self,lon,lat):
        return (MW/2+(lon-self.center[0])*self.cos*self.scale, MH/2-(lat-self.center[1])*self.scale)


def curve(a,b):
    a,b=np.asarray(a),np.asarray(b)
    direction=b-a
    normal=np.array([-direction[1],direction[0]])
    middle=(a+b)/2+normal*.21
    return [(1-t)**2*a+2*t*(1-t)*middle+t*t*b for t in np.linspace(0,1,41)]


class Polyline:
    def __init__(self,points):
        self.points=np.asarray(points,float)
        self.distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
    def at(self,ratio):
        length=max(0,min(1,ratio))*self.distance[-1]
        return tuple(np.interp(length,self.distance,self.points[:,j]) for j in (0,1))


class Story:
    def __init__(self,payload,geodata):
        self.data=payload;self.summary=payload['summary'];self.plan=payload['plan'];self.time=self.summary['timeline_kst']
        self.stations={s['id']:s for s in payload['stations']}
        self.map=LocalMap(payload['bounds'],geodata)
        self.positions={sid:self.map.point(s['lon'],s['lat']) for sid,s in self.stations.items()}
        self.paths=[curve(self.positions[a],self.positions[b]) for a,b,_ in payload['edges']]
        self.trajectory={s['id']:np.asarray(s['trajectory']) for s in self.stations.values()}
        self.departure=datetime.fromisoformat(self.time['departure'])
        self.deliveries=[s for s in self.plan['route']['stops'] if s['action']=='deliver']
        self.receiver=self.stations[self.deliveries[0]['station_id']]
        self.route=self.plan['route']
        road_points=[self.map.point(lon,lat) for lon,lat in self.plan['geometry']['coordinates']]
        self.legs=[];last=0;previous_end=0.
        for stop in self.route['stops']:
            xy=np.asarray(self.map.point(stop['lon'],stop['lat']))
            i=min(range(last,len(road_points)),key=lambda k:np.linalg.norm(np.asarray(road_points[k])-xy))
            self.legs.append((Polyline(road_points[last:i+1]),previous_end,stop['eta_minutes'],stop['completion_minutes']))
            last=i;previous_end=stop['completion_minutes']
        self.legs.append((Polyline(road_points[last:]),previous_end,self.route['return_minutes'],self.route['return_minutes']))

    def spec(self):
        return {'scenes':[
          {'kind':'intro','seconds':13,'title':'타슈 재배치 운영 도우미'},
          {'kind':'history','seconds':20,'title':'대여소 사이의 이용 흐름'},
          {'kind':'flow','seconds':27,'title':'Flow Matching으로 대여·반납 시나리오 계산'},
          {'kind':'risk','seconds':16,'title':'부족 위험과 작업 가능 시간'},
          {'kind':'route','seconds':24,'title':f"재배치 결과 · {self.plan['moved_bikes']}대 이동"},
          {'kind':'result','seconds':20,'title':'수거·배치·복귀 시간표'},
        ]}

    def background(self,scene):
        image=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(image)
        draw.line((56,89,1864,89),fill=EDGE,width=2)
        text(draw,(57,26),'타슈',33,WHITE,True)
        text(draw,(147,34),'재배치 운영 도우미',25,MUTED)
        text(draw,(1863,36),'2026.11.10  ·  관측 기록 재생  ·  KST',23,MUTED,False,'ra')
        text(draw,(57,116),scene['title'],52,WHITE,True)
        draw.rounded_rectangle((PX,MY,PX+PW,MY+MH),radius=19,fill=PANEL,outline=EDGE,width=2)
        return image,draw

    def node(self,draw,station,color=BLUE,radius=6):
        x,y=self.positions[station['id']];bubble(draw,x,y,None,color,radius)

    def history(self,draw,t,muted=False):
        for index,path in enumerate(self.paths):
            points=[tuple(x) for x in path]
            draw.line(points,fill='#274358' if muted else '#367389',width=2 if muted else 3)
            if not muted:
                i=int((t*.57+index*.173)%1*40)
                trail=points[max(0,i-7):i+1]
                if len(trail)>1:draw.line(trail,fill='#55CABD',width=4)
                x,y=points[i];draw.ellipse((x-4,y-4,x+4,y+4),fill=MINT)

    def latent(self,draw,phase):
        step=min(15,int(phase));fraction=phase-step
        for sid,paths in self.trajectory.items():
            cx,cy=self.positions[sid]
            for index,path in enumerate(paths):
                points=[(cx+z[0]*14,cy-z[1]*14) for z in path[max(0,step-4):step+1]]
                x,y=lerp(path[step],path[step+1],fraction)
                end=(cx+x*14,cy-y*14)
                color=(MINT,BLUE,'#B3A1F9','#E3F2FF')[index]
                draw.line(points+[end],fill='#44968F',width=2)
                draw.ellipse((end[0]-3,end[1]-3,end[0]+3,end[1]+3),fill=color)

    def scenario(self,draw,index,t):
        rentals=returns=0
        for s in self.stations.values():
            a,b=s['counts'][index];rentals+=a;returns+=b
            x,y=self.positions[s['id']]
            if a:
                d=10+(t*.9%1)*23
                draw.line((x+9,y-6,x+d,y-d*.65),fill=AMBER,width=4)
                text(draw,(x+22,y-34),f'-{a}',21,AMBER,True)
            if b:
                d=32-(t*.9%1)*23
                draw.line((x-d,y+d*.65,x-9,y+6),fill=MINT,width=4)
                text(draw,(x-27,y+13),f'+{b}',21,MINT,True)
        return rentals,returns

    def result_map(self,draw,elapsed):
        for i,(leg,_,_,_) in enumerate(self.legs):
            points=[tuple(p) for p in leg.points]
            if len(points)>1:
                draw.line(points,fill='#10263D',width=12,joint='curve')
                draw.line(points,fill=BLUE if i==len(self.legs)-1 else MINT,width=6,joint='curve')
                for fraction in (.25,.52,.77):
                    a,b=np.asarray(leg.at(fraction-.008)),np.asarray(leg.at(fraction+.008))
                    direction=b-a
                    if np.linalg.norm(direction)<.01:continue
                    direction=direction/np.linalg.norm(direction)
                    normal=np.array([-direction[1],direction[0]])
                    draw.polygon([tuple(b+direction*7),tuple(b-direction*6+normal*5),tuple(b-direction*6-normal*5)],fill=WHITE)
        for i,stop in enumerate(self.route['stops'],1):
            x,y=self.map.point(stop['lon'],stop['lat'])
            color=MINT if stop['action']=='pickup' else AMBER
            bubble(draw,x,y,i,color,20)
            label(draw,x+27,y-47,('수거' if stop['action']=='pickup' else '배치')+f" {stop['bikes']}대",color)
        if elapsed>=self.route['return_minutes']-1e-6:return
        for leg,start,arrival,complete in self.legs:
            if elapsed<=complete+1e-6:
                fraction=1 if arrival<=start else (elapsed-start)/(arrival-start)
                x,y=leg.at(fraction)
                draw.ellipse((x-10,y-10,x+10,y+10),fill=WHITE,outline=BG,width=3)
                break

    def result_panel(self,draw,elapsed,show_clock):
        x=PX+34
        text(draw,(x,MY+32),f"{self.plan['moved_bikes']}대",69,MINT,True)
        text(draw,(x+171,MY+44),f"{len(self.route['stops'])}곳 방문",29,WHITE,True)
        seconds=round(self.route['return_minutes']*60)
        text(draw,(x,MY+122),f"왕복 {self.route['distance_meters']/1000:.2f}km  ·  {seconds//60}분 {seconds%60:02d}초",29,MUTED)
        draw.line((x,MY+178,PX+PW-35,MY+178),fill=EDGE,width=2)
        rows=[(0.,'작업 시작','버드내초등학교')]
        rows += [(s['completion_minutes'],('수거' if s['action']=='pickup' else '배치')+f" {s['bikes']}대",s['name'].replace('태평동 ','').replace('문화동 ','')) for s in self.route['stops']]
        rows.append((self.route['return_minutes'],'차량 복귀','버드내초등학교'))
        for index,(when,action,place) in enumerate(rows):
            y=MY+204+index*94
            done=elapsed+1e-7>=when
            shade=MINT if done else MUTED
            text(draw,(x,y),clock(self.departure+timedelta(minutes=when)),27,shade,True)
            text(draw,(x+179,y),action,27,WHITE,True)
            text(draw,(x+179,y+35),place,22,MUTED)
        if show_clock:
            text(draw,(x,MY+622),'계획 시각',23,MUTED)
            text(draw,(x+155,MY+611),clock(self.departure+timedelta(minutes=elapsed)),34,WHITE,True)
        else:
            per_bike=round(self.plan['operations']['work_minutes_per_delivered_bike']*60)
            text(draw,(x,MY+621),f'이동 1대당 {per_bike//60}분 {per_bike%60:02d}초 · 왕복 포함',24,MUTED)

    def render(self,scene,t):
        image,draw=self.background(scene);map_image=self.map.base.copy();md=ImageDraw.Draw(map_image)
        kind=scene['kind'];x=PX+34
        if kind in ('intro','history','flow','risk'):
            if kind in ('history','flow'):self.history(md,t,kind=='flow')
            for s in self.stations.values():
                p=s['forecast']['p0'];self.node(md,s,RED if kind=='risk' and p>=.5 else BLUE,6+int(7*p) if kind=='risk' else 6)
        if kind=='intro':
            text(draw,(x,MY+34),'운영 기능',34,WHITE,True)
            features=[('재고와 부족 위험 예측','대여소별로 자전거가 없을 가능성 확인'),('묶음 수거·배치','어디에서 몇 대를 가져올지 계산'),
                      ('차량 경로와 작업표','방문 순서·주행·상하차 시간 계산'),('계획 질문','전체 재배치 계획을 한 번에 확인')]
            for i,(name,desc) in enumerate(features):
                y=MY+112+i*132
                draw.rounded_rectangle((x,y+8,x+12,y+35),radius=5,fill=(MINT,AMBER,BLUE,'#B3A1F9')[i])
                text(draw,(x+28,y),name,30,WHITE,True);text(draw,(x+28,y+46),desc,22,MUTED)
            label(md,19,19,'대전 태평동 · 문화동 일대')
        elif kind=='history':
            label(md,19,19,'과거 이용 기록의 대여소 연결',MINT)
            text(draw,(x,MY+38),'GCN 입력',34,BLUE,True)
            text(draw,(x,MY+110),f"대여소 {len(self.stations)}곳",49,WHITE,True)
            text(draw,(x,MY+185),f"과거 연결 {len(self.paths)}개",34,MINT,True)
            lines(draw,x,MY+285,['주변 대여소의 이용 정보를','각 대여소의 예측에 반영합니다.'],27)
            lines(draw,x,MY+503,['이 지도에 표시한 구간의 연결입니다.','점은 연결을 따라 움직이도록 표시했습니다.'],23)
        elif kind=='flow':
            text(draw,(x,MY+33),'Flow Matching',36,MINT,True)
            lines(draw,x,MY+100,['GCN의 대여소 정보','포아송 대여·반납 평균'],26)
            draw.line((x,MY+195,PX+PW-35,MY+195),fill=EDGE,width=2)
            if t<14:
                phase=16*min(1,max(0,(t-.7)/12.5));self.latent(md,phase)
                label(md,19,19,'대여소별 대여·반납 보정값 이동',MINT)
                text(draw,(x,MY+233),'가우시안 잡음에서 시작',29,WHITE,True)
                text(draw,(x,MY+294),f'{int(phase):02d} / 16',69,MINT,True)
                text(draw,(x,MY+387),'표본을 만드는 단계',27,MUTED)
                draw.rounded_rectangle((x,MY+449,x+465,MY+460),radius=5,fill=EDGE)
                if phase>0:draw.rounded_rectangle((x,MY+449,x+465*phase/16,MY+460),radius=5,fill=MINT)
            else:
                sample=min(3,int((t-14)/3.25));a,b=self.scenario(md,sample,t)
                label(md,19,19,f'대여소별 예측 표본 {sample+1} / 4',MINT)
                text(draw,(x,MY+233),f'표시한 {len(self.stations)}곳의 표본',29,WHITE,True)
                text(draw,(x,MY+307),f'대여 {a}건',43,AMBER,True)
                text(draw,(x,MY+384),f'반납 {b}건',43,MINT,True)
                text(draw,(x,MY+478),'재고 = 현재 재고 - 대여 + 반납',26,WHITE)
            lines(draw,x,MY+571,['대여소마다 128개 표본을 계산합니다.','지도에는 표본 일부를 표시합니다.'],22)
        elif kind=='risk':
            s=self.receiver;x0,y0=self.positions[s['id']];p=100*s['forecast']['p0']
            bubble(md,x0,y0,None,RED,23);label(md,x0+27,y0-43,f'센트럴파크  ·  0대 확률 {p:.0f}%',RED)
            label(md,19,19,'21:00 재고가 0대일 확률 · 50% 이상 강조',RED)
            text(draw,(x,MY+31),'센트럴파크(105동)',34,WHITE,True)
            text(draw,(x,MY+95),f'{p:.0f}%',72,RED,True)
            text(draw,(x,MY+188),'21:00에 자전거가 없을 확률',25,MUTED)
            draw.line((x,MY+247,PX+PW-35,MY+247),fill=EDGE,width=2)
            windows=[('관측',clock(self.time['observation'],False)),('수거·배치 마감',clock(self.time['cutoff'])),
                     ('예측 대상',clock(self.time['target'],False)),('차량 복귀 마감',clock(self.time['return_deadline']))]
            for i,(label_text,value) in enumerate(windows):
                y=MY+276+i*66;text(draw,(x,y),label_text,26,MUTED);text(draw,(PX+PW-39,y),value,29,WHITE,True,'ra')
            lines(draw,x,MY+565,['최소 3대/곳 · 정차 준비 1분','상하차 20초/대 · 출발 후 15분 복귀'],23)
        else:
            total=self.route['return_minutes']
            elapsed=total if kind=='result' else total*min(1,t/(scene['seconds']-3))
            self.result_map(md,elapsed);self.result_panel(draw,elapsed,kind=='route')
            label(md,19,19,'계산된 차량 왕복 경로',MINT)
        md.rectangle((0,MH-36,MW,MH),fill='#101D2D')
        text(md,(15,MH-30),'© OpenStreetMap contributors',18,MUTED)
        text(md,(MW-14,MH-30),'대전 태평동 · 문화동',18,MUTED,False,'ra')
        image.paste(map_image,(MX,MY))
        if kind in ('route','result'):
            ops=self.plan['operations']
            travel=round((self.route['return_minutes']-ops['setup_minutes']-ops['handling_minutes'])*60)
            text(draw,(MX,960),f"정차 준비 {ops['setup_minutes']:g}분   +   상하차 {ops['handling_minutes']:g}분   +   도로 주행 {travel//60}분 {travel%60:02d}초",29,WHITE)
            draw.line((1395,980,1436,980),fill=MINT,width=5)
            text(draw,(1449,963),'수거·배치 경로',23,MUTED)
            draw.line((1649,980,1690,980),fill=BLUE,width=5)
            text(draw,(1703,963),'복귀 경로',23,MUTED)
        else:
            stages=['과거 이용 연결','대여·반납 시나리오','부족 위험','차량 작업 계획']
            active={'intro':-1,'history':0,'flow':1,'risk':2}[kind]
            for index,stage in enumerate(stages):
                start=MX+index*385
                text(draw,(start,957),stage,25,MINT if index==active else MUTED,index==active)
                if index<3:text(draw,(start+288,957),'→',26,MUTED)
        return image


def encode(story,spec,output,ffmpeg):
    offset=0;timeline=[];clips=[]
    for i,scene in enumerate(spec['scenes']):
        duration=scene['seconds']
        clip=output/f'clip-{i}.mp4';clips.append(clip)
        static=scene['kind'] in ('intro','risk','result')
        rate=1 if static else FPS
        cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','image2pipe','-framerate',str(rate),'-vcodec','mjpeg','-i','pipe:0',
             '-vf','fps=25,tpad=stop_mode=clone:stop_duration=1','-t',str(duration),'-an','-c:v','libx264','-threads','4',
             '-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(clip)]
        print(f'Rendering scene {i+1}: {scene["kind"]}',flush=True)
        with subprocess.Popen(cmd,cwd=output,stdin=subprocess.PIPE) as process:
            base=story.render(scene,0) if static else None
            if static:
                encoded=BytesIO();base.save(encoded,'JPEG',quality=94,subsampling=0);still=encoded.getvalue()
            for j in range(duration*rate):
                if static:frame=still
                else:
                    encoded=BytesIO();story.render(scene,j/rate).save(encoded,'JPEG',quality=94,subsampling=0);frame=encoded.getvalue()
                process.stdin.write(frame)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('Scene encoding failed')
        timeline.append(dict(kind=scene['kind'],start=offset,end=offset+duration,title=scene['title']))
        offset+=duration
    concat=output/'concat.txt';concat.write_text('\n'.join(f"file '{clip.name}'" for clip in clips),encoding='utf-8')
    movie=output/'tashu-map-scenario-2min.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(movie)],check=True)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(movie),'-f','null','-'],check=True)
    manifest=dict(duration_seconds=offset,resolution=[WIDTH,HEIGHT],timeline=timeline,sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
                  bytes=movie.stat().st_size,decoded=True,render_host='dislab',audio_streams=0,result_only_after_seconds=76)
    (output/'video-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ffmpeg');parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    story=Story(json.loads((args.data/'story.private.json').read_text(encoding='utf-8')),json.loads((args.data/'map.osm.json').read_bytes()))
    spec=story.spec()
    if args.prepare:
        (output/'scene-timeline.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for i,(scene,t) in enumerate(zip(spec['scenes'],[0,8,8,0,16,0])):
            story.render(scene,t).save(output/f'preview-{i}.png')
        story.render(spec['scenes'][2],20).save(output/'preview-scenario.png')
    else:
        encode(story,spec,output,args.ffmpeg)


if __name__=='__main__':main()
