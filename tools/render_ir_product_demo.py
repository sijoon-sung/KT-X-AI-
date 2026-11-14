"""Produce the 90-second IR cut from the route calculation and GPT captures."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import render_map_scenario_video as v
from render_dispatch_scenario_video import ScenarioStory


BG = '#0A1522'
CARD = '#111F2F'
RULE = '#30465B'
COLORS = (v.MINT, v.BLUE, v.AMBER)
FLOW_STATION = 'ST1155'


def ease(t):
    t = max(0, min(1, t))
    return t*t*(3-2*t)


def panel(draw, box, fill=CARD):
    draw.rounded_rectangle(box, radius=23, fill=fill, outline=RULE, width=2)


def pill(draw, x, y, text, color=v.MINT):
    w = draw.textlength(text, font=v.font(24, True)) + 34
    draw.rounded_rectangle((x, y, x+w, y+45), radius=22, fill='#152D3B')
    v.text(draw, (x+17, y+4), text, 24, color, True)
    return w


def section(title, stage):
    image = Image.new('RGB', (v.WIDTH, v.HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    v.text(draw, (66, 33), '타슈 재배치', 29, v.MINT, True)
    v.text(draw, (1854, 38), stage, 24, v.MUTED, True, 'ra')
    draw.line((65, 88, 1855, 88), fill=RULE, width=2)
    v.text(draw, (64, 110), title, 53, v.WHITE, True)
    return image, draw


def place(frame, image, x, y, w, h):
    image = ImageOps.fit(image, (w, h), Image.Resampling.LANCZOS)
    mask = Image.new('L', image.size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w-1, h-1), radius=22, fill=255)
    frame.paste(image, (x, y), mask)


class IR(ScenarioStory):
    def __init__(self, local, geodata, network, cases, review, comparison, roads, shots):
        super().__init__(local, geodata, network, cases, review, comparison)
        self.fleet_roads = v.LocalMap(roads['bounds'], roads['map']) if roads else None
        self.flow_station = self.stations[FLOW_STATION]
        scenario = self.flow_station
        self.flow_samples = list(dict.fromkeys(
            (rent, returned, stock) for (rent, returned), stock in
            zip(scenario['counts'], scenario['stock_samples'], strict=True)
        ))
        if len(self.flow_samples) != 3 or any(
            max(0, scenario['current'] - rent + returned) != stock
            for rent, returned, stock in self.flow_samples
        ):
            raise ValueError('Unexpected archived flow examples for the selected station')
        self.shots = {}
        for name in ('before', 'calculating', 'fleet', 'question', 'pending', 'answer'):
            with Image.open(shots / f'{name}.png') as image:
                if image.size != (1280, 720):
                    raise ValueError(f'Capture size changed: {name} {image.size}')
                self.shots[name] = image.convert('RGB')

    def spec(self):
        rows = [
            ('pitch', '자전거가 떨어지기 전에', 8, False),
            ('zoom', '대전 전체에서 작업 권역까지', 8, False),
            ('flow', '반납으로 들어오고, 대여로 나갑니다', 11, False),
            ('risk', '먼저 채울 대여소를 찾습니다', 6, True),
            ('before', '차량 경로를 계산합니다', 5, True),
            ('calculating', '차량 경로를 계산합니다', 1, True),
            ('fleet', '세 권역에 차량 작업을 나눕니다', 9, False),
            ('district', '수거부터 복귀까지', 10, False),
            ('question', 'GPT에게 작업 순서를 묻습니다', 6, True),
            ('pending', 'GPT에게 작업 순서를 묻습니다', 3, True),
            ('answer', 'GPT가 작업표를 읽어 답합니다', 11, True),
            ('result', '저녁 배차 작업표', 12, True),
        ]
        return {'scenes': [dict(kind=k, title=t, seconds=s, static=f) for k,t,s,f in rows]}

    def pitch(self, t):
        frame = Image.new('RGB', (v.WIDTH, v.HEIGHT), BG)
        draw = ImageDraw.Draw(frame)
        city, _ = self.city_map(t)
        place(frame, city, 778, 68, 1076, 944)
        pill(draw, 66, 127, '대전 공공자전거')
        v.text(draw, (62, 226), '타슈 재배치', 86, v.WHITE, True)
        for i, line in enumerate(('자전거가 떨어지기 전에', '어디를 채울지 정합니다')):
            v.text(draw, (66, 407+i*78), line, 49, v.WHITE, True)
        draw.line((66, 640, 682, 640), fill=RULE, width=2)
        v.text(draw, (66, 682), '공공자전거 운영팀을 위한 배차 도구', 30, v.MUTED)
        v.text(draw, (66, 953), '저녁 배차 시연 · 관측 기록 재생', 23, v.MUTED)
        v.text(draw, (1818, 978), '© OpenStreetMap contributors', 19, v.MUTED, anchor='ra')
        return frame

    def forecast(self, scene, t):
        kind = scene['kind']
        frame, draw = section(scene['title'], '부족 예측')
        if kind == 'flow':
            return self.station_flows(frame, draw, t)
        if kind == 'zoom':
            city, _ = self.city_map(t, ease((t-1)/6))
            small, detail = '대전 전체 이용 연결', '태평·문화권'
        else:
            city = self.map.base.copy()
            md = ImageDraw.Draw(city)
            for station in self.stations.values():
                self.node(md, station)
            x, y = self.positions[self.receiver['id']]
            md.ellipse((x-31, y-31, x+31, y+31), outline=v.RED, width=2)
            v.bubble(md, x, y, None, v.RED, 15)
            v.label(md, x+19, y-57, '센트럴파크', v.RED)
            small, detail = '부족 위험', '문화동 센트럴파크'
        place(frame, city, 64, 216, 1275, 736)
        panel(draw, (1365, 216, 1854, 952))
        pill(draw, 1402, 259, small, v.RED if kind == 'risk' else v.MINT)
        v.text(draw, (1400, 342), detail, 38, v.WHITE, True)
        if kind == 'risk':
            v.lines(draw, 1402, 444, ['저녁 배차에서', '먼저 채울 곳입니다.'], 29)
        else:
            v.lines(draw, 1402, 444, ['대여소 사이의', '이용 연결을 봅니다.'], 29)
        v.text(draw, (66, 974), '© OpenStreetMap contributors', 20, v.MUTED)
        return frame

    def station_flows(self, frame, draw, t):
        city = self.map.base.copy()
        md = ImageDraw.Draw(city)
        station = self.flow_station
        x, y = self.positions[station['id']]
        v.bubble(md, x, y, None, v.MINT, 21)
        place(frame, city, 64, 216, 789, 736)
        pill(draw, 91, 244, '태평동 쌍용예가')

        panel(draw, (880, 216, 1854, 952))
        pill(draw, 918, 247, 'GCN + Flow Matching')
        v.text(draw, (1815, 255), '같은 관측에서 나온 예측 예시', 22, v.MUTED, anchor='ra')
        # Hold each joint sample before changing the highlighted row.
        selected = min(int(t/2.45), len(self.flow_samples)-1)
        rent, returned, _ = self.flow_samples[selected]
        phase = (t*.7) % 1
        v.text(draw, (1047, 337), '반납 · 들어오는 양', 28, v.MINT, True, 'ma')
        v.text(draw, (1674, 337), '대여 · 나가는 양', 28, v.AMBER, True, 'ma')
        panel(draw, (1250, 339, 1482, 506), fill='#162D40')
        v.text(draw, (1366, 375), f"{station['current']}대", 49, v.WHITE, True, 'ma')
        v.text(draw, (1366, 447), '관측 재고', 23, v.MUTED, anchor='ma')
        for left, right, count, color in ((942,1224,returned,v.MINT), (1519,1801,rent,v.AMBER)):
            y = 431
            line = color if count else RULE
            draw.line((left, y, right, y), fill=line, width=4+count*4)
            draw.polygon(((right,y), (right-22,y-15), (right-22,y+15)), fill=line)
            for index in range(count):
                cx = v.lerp(left+12, right-26, ((index/count)+phase) % 1)
                v.bubble(draw, cx, y, None, color, 10)
            v.text(draw, ((left+right)//2, 461), f'{count}대', 29, color, True, 'ma')
        cols = (958, 1162, 1373, 1730)
        for x, title in zip(cols, ('예시','반납','대여','10분 뒤 재고'), strict=True):
            v.text(draw, (x, 555), title, 24, v.MUTED, True, 'ma')
        for index, (outgoing, incoming, stock) in enumerate(self.flow_samples):
            y = 601+index*99
            focused = index == selected
            draw.rounded_rectangle((915, y, 1819, y+81), radius=14,
                                   fill='#193448' if focused else '#142437',
                                   outline=v.BLUE if focused else RULE, width=2)
            for x, label, color in zip(
                cols, (str(index+1), f'{incoming}대', f'{outgoing}대', f'{stock}대'),
                (v.MUTED, v.MINT, v.AMBER, v.WHITE), strict=True,
            ):
                v.text(draw, (x, y+20), label, 33, color, True, 'ma')
        v.text(draw, (66, 974), '© OpenStreetMap contributors', 20, v.MUTED)
        return frame

    def screen(self, scene):
        frame, draw = section(scene['title'], '경로 계산')
        place(frame, self.shots[scene['kind']], 224, 209, 1472, 828)
        draw.rounded_rectangle((223, 208, 1696, 1037), radius=22, outline=RULE, width=2)
        if scene['kind'] == 'before':
            draw.rounded_rectangle((249, 354, 369, 415), radius=10, outline=v.MINT, width=3)
        return frame

    def street_route(self, t):
        base = self.map.base.copy()
        draw = ImageDraw.Draw(base)
        case = self.fleet[0]
        pts = [self.map.point(*p) for p in case['geometry']['coordinates']]
        draw.line(pts, fill='#091828', width=13, joint='curve')
        draw.line(pts, fill=v.MINT, width=6, joint='curve')
        if t is not None:
            x, y = v.Polyline(pts).at(ease((t-.5)/8))
            v.bubble(draw, x, y, None, v.WHITE, 10)
        for i, stop in enumerate(case['stops'], 1):
            x, y = self.map.point(stop['lon'], stop['lat'])
            color = v.MINT if i == 1 else v.AMBER
            v.bubble(draw, x, y, i, color, 15)
            v.label(draw, x+18, y-50, '수거' if i == 1 else '배치', color)
        return base

    def route_board(self, scene, t):
        kind = scene['kind']
        district = kind == 'district'
        final = kind == 'result'
        frame, draw = section(scene['title'], '재배치 결과' if final else '경로 계산')
        if district:
            route_map = self.street_route(t)
        else:
            route_map = self.fleet_map(None if final else t, self.fleet_roads, show_quantities=False)
        place(frame, route_map, 64, 215, 1135, 700)
        v.text(draw, (65, 934), '© OpenStreetMap contributors', 19, v.MUTED)
        v.text(draw, (1198, 934), '차량 왕복 경로', 20, v.MUTED, anchor='ra')
        if district:
            case = self.fleet[0]
            panel(draw, (1232, 215, 1854, 915))
            pill(draw, 1262, 252, '태평·문화')
            rows = [(s['action'], s['name'], str(s['bikes'])+'대') for s in case['stops']]
            rows.append(('return', case['depot']['name'], ''))
            for i,(action,name,qty) in enumerate(rows):
                y = 348+i*154
                word,color = {'pickup':('수거',v.MINT),'deliver':('배치',v.AMBER),'return':('복귀',v.BLUE)}[action]
                v.text(draw,(1262,y),word,27,color,True)
                v.text(draw,(1818,y-5),qty,35,color,True,'ra')
                v.text(draw,(1262,y+48),name,32,v.WHITE,True)
                if i < 2:
                    draw.line((1262,y+110,1818,y+110), fill=RULE, width=2)
        else:
            for i, case in enumerate(self.fleet):
                y = 215+i*241
                color = COLORS[i]
                panel(draw, (1232, y, 1854, y+218))
                v.text(draw, (1265, y+19), case['name'].split()[0], 34, color, True)
                v.text(draw, (1821, y+16), str(case['moved_bikes'])+'대', 40, color, True, 'ra')
                for j, stop in enumerate(case['stops']):
                    row = y+88+j*54
                    v.text(draw, (1265,row), '수거' if j == 0 else '배치', 24, v.MUTED, True)
                    v.text(draw, (1344,row-1), stop['name'], 27, v.WHITE, True)
            v.text(draw, (1234, 935), '차량별로 각 권역에서 출발하는 조건', 22, v.MUTED)
        return frame

    def conversation(self, scene):
        kind = scene['kind']
        frame, draw = section(scene['title'], '출발 전 확인')
        place(frame, self.street_route(None), 64, 215, 915, 583)
        pill(draw, 88, 239, '태평·문화 차량')
        v.text(draw, (66, 822), '© OpenStreetMap contributors', 19, v.MUTED)
        crop = self.shots[kind].crop((752, 188, 1257, 681))
        place(frame, crop, 1011, 215, 843, 822)
        draw.rounded_rectangle((1011, 215, 1854, 1037), radius=22, outline=RULE, width=2)
        if kind == 'question':
            draw.rounded_rectangle((1673, 638, 1813, 718), radius=12, outline=v.MINT, width=3)
        return frame

    def render(self, scene, t):
        kind = scene['kind']
        if kind == 'pitch':
            return self.pitch(t)
        if kind in ('zoom', 'flow', 'risk'):
            return self.forecast(scene,t)
        if kind in ('before', 'calculating'):
            return self.screen(scene)
        if kind == 'fleet' and t < 6:
            screen = self.screen(scene)
            if t < 5.4:
                return screen
            return Image.blend(screen, self.route_board(scene, t-5.4), ease((t-5.4)/.6))
        if kind in ('fleet', 'district', 'result'):
            return self.route_board(scene,t)
        return self.conversation(scene)


def main():
    parser = argparse.ArgumentParser()
    for name in ('local','scenarios','shots','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)

    def read(path):
        return json.loads(path.read_bytes())

    map_path = args.scenarios/'ir-fleet-map.json'
    story = IR(read(args.local/'story.private.json'),read(args.local/'map.osm.json'),
               read(args.scenarios/'city-network.json'),read(args.scenarios/'plans.json')['cases'],
               read(args.scenarios/'review.json'),read(args.scenarios/'comparison-map.json'),
               read(map_path) if map_path.exists() else None,args.shots)
    if len(story.fleet) != 3:
        raise ValueError('The demo requires three checked routes')
    v.encode(story,story.spec(),args.out.resolve(),args.ffmpeg)


if __name__ == '__main__':
    main()
