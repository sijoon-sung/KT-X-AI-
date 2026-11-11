"""Video frames of the traced residual states, with linear interpolation for display."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def frames(trace_file, destination, seconds, fps=8):
    trace = json.loads(Path(trace_file).read_text(encoding='utf-8'))
    path = np.asarray(trace['trajectory'])
    steps = trace['integration_steps']
    count = trace['samples']
    if path.shape != (steps + 1, count, 2) or not trace['exact_original_integrator']:
        raise ValueError('Expected an original-sampler trace')
    limit = max(3, int(np.ceil(np.abs(path).max())))
    navy, teal, muted = '#173E50', '#087F80', '#566D78'
    fonts = {}

    def font(size, bold=False):
        key = (size, bold)
        if key not in fonts:
            fonts[key] = ImageFont.truetype('C:/Windows/Fonts/' + ('malgunbd.ttf' if bold else 'malgun.ttf'), size)
        return fonts[key]

    left, top, right, bottom = 117, 273, 1148, 817
    def pixels(xy):
        return (left + (xy[0] + limit) / (2 * limit) * (right - left),
                bottom - (xy[1] + limit) / (2 * limit) * (bottom - top))

    base = Image.new('RGB', (1920, 1080), '#F3F7F9')
    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 0, 1920, 98), fill=navy)
    draw.text((56, 24), '타슈  /  운영 도우미', font=font(32, True), fill='white')
    draw.text((1190, 31), '고정 모델 추론 · 합성 대여소 조건', font=font(26), fill='#D7E7EA')
    draw.text((74, 132), 'Flow Matching으로 가능한 이용량 만들기', font=font(53, True), fill=navy)
    draw.text((77, 204), '대여·반납 보정값의 움직임', font=font(29), fill=muted)
    draw.rounded_rectangle((78, 253, 1187, 903), radius=22, fill='white', outline='#D7E3E9', width=2)
    for tick in range(-limit, limit + 1):
        x = pixels((tick, 0))[0]
        y = pixels((0, tick))[1]
        color = '#9BB2BD' if tick == 0 else '#E7EDF1'
        draw.line((x, top, x, bottom), fill=color, width=2)
        draw.line((left, y, right, y), fill=color, width=2)
    draw.text((left + 7, top + 9), '반납 보정값 ↑', font=font(24, True), fill=muted)
    draw.text((right - 200, bottom + 20), '대여 보정값 →', font=font(24, True), fill=muted)
    draw.rounded_rectangle((1230, 253, 1840, 903), radius=22, fill='#E6F3F2')
    draw.text((1280, 285), '가우시안 잡음에서 시작', font=font(34, True), fill=navy)
    draw.text((1280, 346), '대여소 관계(GCN)와 포아송 평균을 참고', font=font(25), fill=muted)
    draw.text((1280, 474), '점 하나 = 가능한 대여·반납 한 쌍', font=font(28, True), fill=navy)
    draw.rectangle((0, 934, 1920, 1080), fill=navy)

    for frame in range(seconds * fps):
        t = frame / fps
        phase = np.clip((t - 3.5) / 13.5, 0, 1) * steps
        step = min(int(phase), steps - 1)
        fraction = phase - step
        current = (1 - fraction) * path[step] + fraction * path[step + 1]
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for index, point in enumerate(current):
            if step:
                trail = [pixels(p) for p in path[max(0, step - 3):step + 1, index]]
                draw.line(trail + [pixels(point)], fill='#ACD5D5', width=2)
            x, y = pixels(point)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=teal)
        draw.text((1280, 396), f'생성 단계 {int(phase)} / {steps}', font=font(40, True), fill=teal)
        draw.rounded_rectangle((1280, 552, 1784, 568), radius=8, fill='#C4DFDD')
        if phase > 0:
            draw.rounded_rectangle((1280, 552, 1280 + 504 * phase / steps, 568), radius=8, fill=teal)
        if phase >= steps:
            draw.text((1280, 606), '현재 2대와 합친 재고 표본', font=font(28, True), fill=navy)
            histogram = sorted((int(k), v) for k, v in trace['stock_histogram'].items())
            height = 124
            base_y = 796
            band = 470 / len(histogram)
            for index, (stock, occurrences) in enumerate(histogram):
                x = 1290 + index * band
                bar = height * occurrences / max(v for _, v in histogram)
                draw.rounded_rectangle((x, base_y - bar, x + band * .65, base_y), radius=4, fill=teal)
                draw.text((x, base_y + 8), f'{stock}대', font=font(22), fill=muted)
            draw.text((1280, 854), f'가능한 경우 {count}개', font=font(25), fill=muted)
        else:
            draw.text((1280, 621), '마지막 점을 대여·반납 건수로 바꾸고', font=font(28), fill=muted)
            draw.text((1280, 670), '현재 재고에 더하거나 뺍니다.', font=font(28), fill=muted)
        image.save(str(destination).replace('%03d', f'{frame:03d}'), quality=91, subsampling=0)
    return fps
