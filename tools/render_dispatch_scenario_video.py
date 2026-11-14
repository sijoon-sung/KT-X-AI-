"""Show city flow, a continuous zoom, then the independently checked road plans."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

import render_map_scenario_video as v


class ScenarioStory(v.Story):
    def __init__(self,local,geodata,network,cases,review,comparison):
        super().__init__(local,geodata)
        self.network=network
        self.cases={c['id']:c for c in cases}
        self.review=review
        self.fleet=[self.cases[k] for k in review['fleet']]
        self.comparison_map=v.LocalMap(comparison['bounds'],comparison['map'])
        self.city_center=(127.40,36.346)
        self.city_scale=1950

    def spec(self):
        return {'scenes':[
            dict(kind='intro',title='타슈 재배치 운영 도우미',seconds=8),
            dict(kind='city',title='대전 전체 대여소의 이용 연결',seconds=13),
            dict(kind='zoom',title='도시 전체에서 작업 권역으로',seconds=9),
            dict(kind='history',title='태평·문화권의 대여소 연결',seconds=10),
            dict(kind='flow',title='Flow Matching으로 대여·반납 시나리오 계산',seconds=27),
            dict(kind='risk',title='부족 위험을 작업 계획에 반영',seconds=9),
            dict(kind='fleet',title='같은 관측에서 세 권역의 경로 계산',seconds=16),
            dict(kind='alternatives',title='권역과 작업 조건에 따라 바뀌는 계획',seconds=14),
            dict(kind='review',title='경로 검증과 출발 전 확인 사항',seconds=11),
            dict(kind='result',title='태평·문화권 · 수거·배치·복귀 시간표',seconds=13),
        ]}

    def city_points(self,ratio=0):
        target=self.map.center
        center=tuple(v.lerp(a,b,ratio) for a,b in zip(self.city_center,target))
        scale=math.exp(v.lerp(math.log(self.city_scale),math.log(self.map.scale),ratio))
        return lambda lon,lat:(v.MW/2+(lon-center[0])*self.map.cos*scale,v.MH/2-(lat-center[1])*scale)

    def city_map(self,t,zoom=0,with_edges=True):
        image=Image.new('RGB',(v.MW,v.MH),'#142236');draw=ImageDraw.Draw(image)
        pt=self.city_points(zoom)
        for way in self.network['boundary']:
            draw.line([pt(*p) for p in way],fill='#496580',width=3)
        if with_edges:
            for i,(a,b) in enumerate(self.network['edges']):
                ax,ay=pt(*a);bx,by=pt(*b)
                if max(ax,bx)<0 or min(ax,bx)>v.MW or max(ay,by)<0 or min(ay,by)>v.MH:continue
                draw.line((ax,ay,bx,by),fill='#2D5968',width=1)
                if i%7==0:
                    drift=(t*.43+i*.089)%1
                    x,y=v.lerp(ax,bx,drift),v.lerp(ay,by,drift)
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=v.MINT)
            for position in self.network['stations']:
                x,y=pt(*position)
                if 0<x<v.MW and 0<y<v.MH:draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill='#77A8C3')
        places=[('유성',127.315,36.355),('둔산',127.3848,36.3504),('대전역',127.4346,36.3326),('태평·문화',127.4009,36.3167),('대청호',127.500,36.438)]
        if zoom<.35:
            for name,lon,lat in places:
                x,y=pt(lon,lat)
                if 40<x<v.MW-150 and 40<y<v.MH-75:v.label(draw,x+10,y-42,name,v.WHITE)
        if zoom>0:
            left,bottom,right,top=self.data['bounds'];x0,y0=pt(left,top);x1,y1=pt(right,bottom)
            draw.rounded_rectangle((x0,y0,x1,y1),radius=9,outline=v.MINT,width=3)
            if zoom>=.75:image=Image.blend(image,self.map.base,(zoom-.75)/.25)
        return image,pt

    def fleet_map(self,t):
        colors=(v.MINT,v.BLUE,v.AMBER)
        # Zoom to the recorded fleet footprints while retaining their real orientation.
        pts=[p for c in self.fleet for p in c['geometry']['coordinates']]
        l,r=min(p[0] for p in pts),max(p[0] for p in pts);b,a=min(p[1] for p in pts),max(p[1] for p in pts)
        scale=min(v.MW*.74/(r-l)/self.map.cos,v.MH*.65/(a-b))
        cx,cy=(l+r)/2,(a+b)/2
        pt=lambda lon,lat:(v.MW/2+(lon-cx)*self.map.cos*scale,v.MH/2-(lat-cy)*scale)
        image=Image.new('RGB',(v.MW,v.MH),'#142236');draw=ImageDraw.Draw(image)
        for way in self.network['boundary']:draw.line([pt(*p) for p in way],fill='#3C556D',width=2)
        for position in self.network['stations']:
            x,y=pt(*position)
            if 0<x<v.MW and 0<y<v.MH:draw.ellipse((x-2,y-2,x+2,y+2),fill='#436079')
        for i,case in enumerate(self.fleet):
            points=[pt(*p) for p in case['geometry']['coordinates']]
            draw.line(points,fill=colors[i],width=6,joint='curve')
            line=v.Polyline(points);x,y=line.at(min(1,max(0,(t-i*.8)/9)))
            v.bubble(draw,x,y,None,v.WHITE,9)
            for j,stop in enumerate(case['stops'],1):
                x,y=pt(stop['lon'],stop['lat']);v.bubble(draw,x,y,j,colors[i],12)
            stop=case['stops'][-1];x,y=pt(stop['lon'],stop['lat']);v.label(draw,x+18,y-51,f"{case['name'].split()[0]} · {case['moved_bikes']}대",colors[i])
        return image

    def footer(self,image,map_image,scope):
        md=ImageDraw.Draw(map_image)
        md.rectangle((0,v.MH-36,v.MW,v.MH),fill='#101D2D')
        v.text(md,(15,v.MH-30),'© OpenStreetMap contributors',18,v.MUTED)
        v.text(md,(v.MW-14,v.MH-30),scope,18,v.MUTED,anchor='ra')
        image.paste(map_image,(v.MX,v.MY))

    def overview(self,scene,t):
        image,draw=self.background(scene);x=v.PX+34
        ratio=0.
        if scene['kind']=='zoom':
            u=min(1,t/(scene['seconds']-.5));ratio=u*u*(3-2*u)
        map_image,pt=self.city_map(t,ratio)
        if scene['kind']=='intro':
            v.text(draw,(x,v.MY+34),'운영 기능',34,v.WHITE,True)
            rows=[('부족 위험 예측','대여·반납 시나리오에서 예상 재고 계산'),('묶음 수거·배치','한 곳에서 최소 3대씩 작업'),
                  ('권역별 경로와 작업표','여러 시나리오의 실제 도로 경로 계산'),('출발 전 검토','시간·적재량·작업 장소 확인')]
            for i,(title,body) in enumerate(rows):
                y=v.MY+105+132*i;v.text(draw,(x,y),title,29,(v.MINT,v.AMBER,v.BLUE,v.WHITE)[i],True);v.text(draw,(x,y+46),body,22,v.MUTED)
        else:
            v.text(draw,(x,v.MY+40),'과거 이용 기록',34,v.BLUE,True)
            v.text(draw,(x,v.MY+130),f"{len(self.network['stations']):,}곳",64,v.WHITE,True)
            v.text(draw,(x,v.MY+220),'모델이 지원하는 대전 대여소',26,v.MUTED)
            v.text(draw,(x,v.MY+335),'태평·문화권으로 확대' if ratio else '과거 연결 일부 표시',29,v.MINT,True)
            v.lines(draw,x,v.MY+456,['GCN이 대여소 사이의','과거 이용 관계를 읽습니다.'],26)
        self.footer(image,map_image,'대전광역시 · 과거 대여소 연결')
        return image

    def result_view(self,scene,t):
        image,draw=self.background(scene);x=v.PX+34;kind=scene['kind']
        if kind=='fleet':
            map_image=self.fleet_map(t)
            v.text(draw,(x,v.MY+28),f"총 {sum(c['moved_bikes'] for c in self.fleet)}대",61,v.MINT,True)
            v.text(draw,(x,v.MY+112),'차량 3대 · 작업 대상 6곳',26,v.MUTED)
            for i,c in enumerate(self.fleet):
                y=v.MY+184+135*i;color=(v.MINT,v.BLUE,v.AMBER)[i]
                v.text(draw,(x,y),c['name'].split()[0],30,color,True)
                v.text(draw,(x,y+48),f"{c['moved_bikes']}대 · {c['distance_km']:.2f}km · {c['work_minutes']:.1f}분",27,v.WHITE)
            v.text(draw,(x,v.MY+627),'세 권역에서 따로 출발하는 가정',21,v.MUTED)
        elif kind=='alternatives':
            map_image=self.comparison_map.base.copy();md=ImageDraw.Draw(map_image)
            keys=['taepyeong-2050','return-8-2050','split-2050'];titles=['복귀 15분','복귀 8분','빈자리 3대 가정']
            selected=min(2,int(t/(scene['seconds']/3)))
            c=self.cases[keys[selected]]
            md.line([self.comparison_map.point(*point) for point in c['geometry']['coordinates']],fill=v.MINT,width=7,joint='curve')
            for i,stop in enumerate(c['stops'],1):
                px,py=self.comparison_map.point(stop['lon'],stop['lat']);color=v.MINT if stop['action']=='pickup' else v.AMBER
                v.bubble(md,px,py,i,color,18);v.label(md,px+23,py-43,('수거' if i==1 else '배치')+f" {stop['bikes']}대",color)
            for i,(key,title) in enumerate(zip(keys,titles)):
                row=self.cases[key];y=v.MY+42+198*i;highlight=i==selected
                if highlight:draw.rounded_rectangle((x-13,y-11,v.PX+v.PW-21,y+146),radius=12,fill='#20374B')
                v.text(draw,(x,y),title,30,v.MINT if highlight else v.MUTED,True)
                v.text(draw,(x,y+48),f"{row['moved_bikes']}대 · {row['distance_km']:.2f}km · {row['work_minutes']:.1f}분",27,v.WHITE)
                place=row['stops'][-1]['name']
                v.text(draw,(x,y+93),place,24,v.MUTED)
        else:
            map_image=self.fleet_map(14)
            total=sum(check is not None for check in self.review['checks'].values())
            v.text(draw,(x,v.MY+36),f'작업표 {total}건 검증 통과',33,v.MINT,True)
            v.text(draw,(x,v.MY+96),'수량 보존 · 적재량 · 작업·복귀 마감',23,v.MUTED)
            title='LLM이 먼저 확인할 항목' if self.review['mode']=='llm_priority' else '출발 전에 확인할 항목'
            v.text(draw,(x,v.MY+184),title,30,v.WHITE,True)
            for i,fact in enumerate(self.review['selected'][:3]):
                key=fact['id'];base=key.rsplit('-',1)[0];case=self.cases.get(base)
                if not case or not key.endswith(('-cutoff','-return')):
                    v.lines(draw,x,v.MY+260+i*120,[fact['text'][j:j+29] for j in range(0,len(fact['text']),29)],21,gap=5)
                    continue
                check=self.review['checks'][base];kind_label='복귀' if key.endswith('-return') else '배치'
                seconds=check['return_slack_seconds' if kind_label=='복귀' else 'work_slack_seconds']
                y=v.MY+260+i*120
                v.text(draw,(x,y),case['name'].split()[0]+(' · 복귀 8분' if base.startswith('return-8') else ''),25,v.WHITE,True)
                v.text(draw,(x,y+44),f'{kind_label} 마감까지 {seconds}초',27,v.AMBER,True)
        self.footer(image,map_image,'기록 재생 · 경로 계산 결과')
        return image

    def render(self,scene,t):
        kind=scene['kind']
        if kind in ('intro','city','zoom'):return self.overview(scene,t)
        if kind in ('fleet','alternatives','review'):return self.result_view(scene,t)
        return super().render(scene,t)


def main():
    p=argparse.ArgumentParser()
    for key in ('local','scenarios','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ffmpeg',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    def read(path):return json.loads(path.read_bytes())
    story=ScenarioStory(read(a.local/'story.private.json'),read(a.local/'map.osm.json'),read(a.scenarios/'city-network.json'),
                        read(a.scenarios/'plans.json')['cases'],read(a.scenarios/'review.json'),read(a.scenarios/'comparison-map.json'))
    if len(story.fleet)!=3:raise ValueError('The three-region display requires disjoint, checked plans')
    v.encode(story,story.spec(),a.out.resolve(),a.ffmpeg)


if __name__=='__main__':main()
