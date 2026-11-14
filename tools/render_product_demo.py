"""Edit verified browser captures and the saved city flow into the pitch demo."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import render_map_scenario_video as v
from render_dispatch_scenario_video import ScenarioStory


class ProductDemo(ScenarioStory):
    capture_names = ('fleet', 'taepyeong', 'review', 'final')
    opening_target = 'fleet'

    def __init__(self, local, geodata, network, cases, review, comparison, shots):
        super().__init__(local, geodata, network, cases, review, comparison)
        self.screens = {}
        for name in self.capture_names:
            with Image.open(shots / f'{name}.png') as shot:
                frame = Image.new('RGB', (v.WIDTH, v.HEIGHT), v.BG)
                screen = ImageOps.contain(shot.convert('RGB'), frame.size, Image.Resampling.LANCZOS)
                frame.paste(screen, ((v.WIDTH-screen.width)//2, (v.HEIGHT-screen.height)//2))
                self.screens[name] = frame

    def spec(self):
        return {'scenes': [
            dict(kind='opening', title='오늘 저녁, 어느 권역부터 채울까요?', seconds=8),
            dict(kind='fleet', title='권역별 재배치 작업 계획', seconds=19),
            dict(kind='taepyeong', title='태평·문화 작업표', seconds=18),
            dict(kind='review', title='대전역 출발 전 확인', seconds=12),
            dict(kind='result', title='권역별 수거·배치 작업표', seconds=13),
        ]}

    def opening(self, scene, t):
        if t >= 6.8:
            return Image.blend(self.opening(scene, 6.79), self.screens[self.opening_target], min(1, (t-6.8)/1.2))
        u = min(1, max(0, (t-2.6)/2.8))
        zoom = u*u*(3-2*u)
        frame, draw = self.background(scene)
        map_image, _ = self.city_map(t, zoom)
        x = v.PX+34
        if zoom >= 1:
            md = ImageDraw.Draw(map_image)
            for station in self.stations.values():
                self.node(md, station)
            self.latent(md, 16*min(1, max(0, (t-5.4)/1.35)))
            v.label(md, 19, 19, '대여·반납 시나리오', v.MINT)
            v.text(draw, (x, v.MY+70), 'Flow Matching', 39, v.MINT, True)
            v.lines(draw, x, v.MY+169, ['예상 대여와 반납을 바탕으로', '부족한 대여소를 찾습니다.'], 26)
            scope = '대전 태평동 · 문화동'
        else:
            v.text(draw, (x, v.MY+70), '대전의 대여소 이용 흐름', 33, v.MINT, True)
            v.lines(draw, x, v.MY+169, ['과거 대여·반납 기록으로', '대여소의 연결을 살펴봅니다.'], 26)
            scope = '대전광역시 · 과거 대여소 연결'
        self.footer(frame, map_image, scope)
        return frame

    def render(self, scene, t):
        kind = scene['kind']
        if kind == 'opening':
            return self.opening(scene, t)
        if kind == 'result':
            return self.screens['final']
        return self.screens[kind]


class LiveProductDemo(ProductDemo):
    capture_names = ('before','calculating','fleet','question','pending','answer','final')
    opening_target = 'before'

    def spec(self):
        return {'scenes': [
            dict(kind='opening', title='대전 이용 흐름과 대여·반납 시나리오', seconds=8),
            dict(kind='before', title='경로 계산 버튼', seconds=3),
            dict(kind='calculating', title='경로 계산 중', seconds=1),
            dict(kind='fleet', title='권역별 경로 계산 결과', seconds=6),
            dict(kind='question', title='태평·문화 작업표에 질문', seconds=4),
            dict(kind='pending', title='외부 모델 답변 대기', seconds=2),
            dict(kind='answer', title='계산된 작업표에 대한 LLM 답변', seconds=8),
            dict(kind='result', title='권역별 수거·배치 작업표', seconds=6),
        ]}


def main():
    parser = argparse.ArgumentParser()
    for name in ('local','scenarios','shots','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True)
    parser.add_argument('--live-capture',action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)

    def read(path):
        return json.loads(path.read_bytes())

    story_type = LiveProductDemo if args.live_capture else ProductDemo
    story = story_type(read(args.local/'story.private.json'),read(args.local/'map.osm.json'),
                        read(args.scenarios/'city-network.json'),read(args.scenarios/'plans.json')['cases'],
                        read(args.scenarios/'review.json'),read(args.scenarios/'comparison-map.json'),args.shots)
    if len(story.fleet) != 3:
        raise ValueError('The demo requires the checked three-region plan')
    v.encode(story,story.spec(),args.out.resolve(),args.ffmpeg)


if __name__ == '__main__':
    main()
