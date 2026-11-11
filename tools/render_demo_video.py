"""Render captured real UI, Korean narration and editable script on dislab."""
import argparse,hashlib,json,subprocess,wave
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps

p=argparse.ArgumentParser();p.add_argument('--shots',required=True);p.add_argument('--audio',required=True);p.add_argument('--script',required=True);p.add_argument('--out',required=True);p.add_argument('--ffmpeg',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shots=Path(a.shots);audio=Path(a.audio)
spec=json.loads(Path(a.script).read_text(encoding='utf-8-sig'))
font='C:/Windows/Fonts/malgun.ttf';bold='C:/Windows/Fonts/malgunbd.ttf'
def f(n,b=False):return ImageFont.truetype(bold if b else font,n)
def wrap(text,draw,font,width):
    result=[];line=''
    for paragraph in text.split('\n'):
        line=''
        for word in paragraph.split():
            proposed=' '.join(filter(None,(line,word)))
            if line and draw.textlength(proposed,font=font)>width:
                result.append(line)
                line=word
            else:
                line=proposed
        result.append(line)
    return result
def text(draw,value,xy,font,color,width,spacing=15):
    x,y=xy
    for line in wrap(value,draw,font,width):draw.text((x,y),line,font=font,fill=color);y+=font.size+spacing
    return y
def frame(i,scene,shot):
    im=Image.new('RGB',(1920,1080),'#F3F7F9');d=ImageDraw.Draw(im)
    d.rectangle((0,0,1920,98),fill='#173E50');d.text((56,24),'타슈  /  운영 도우미',font=f(32,True),fill='white')
    d.text((1040,31),'2026.11.10 20:50 관측 기록 재생',font=f(25),fill='#D7E7EA')
    d.text((59,150),f'{i+1:02d}  /  {len(spec["scenes"]):02d}',font=f(28,True),fill='#188584')
    if scene['kind'] in ('diagram',):
        pic=ImageOps.contain(Image.open(shot).convert('RGB'),(1750,796),Image.Resampling.LANCZOS)
        im.paste(pic,((1920-pic.width)//2,112));ImageDraw.Draw(im).rectangle((0,934,1920,1080),fill='#173E50');return im
    y=text(d,scene['title'],(57,220),f(57,True),'#173E50',570,20)
    if 'cards' in scene:
        for index, (label, value) in enumerate(scene['cards']):
            yy=y+45+index*115
            if label:d.text((61,yy),label,font=f(27),fill='#526A75')
            text(d,value,(61,yy+34),f(37,True),'#188584',585,12)
    elif scene['kind']=='cover':
        text(d,'이용 흐름·부족 위험\n수거·배치 수량과 동선\n운영자가 묻고 확인',(60,y+60),f(34),'#526A75',535,24)
    elif scene['kind']=='route':
        for n,v in enumerate([('옮길 자전거','5대'),('총 이동 거리','5.26km'),('예상 소요 시간','9.1분')]):
            yy=y+45+n*115;d.text((61,yy),v[0],font=f(27),fill='#526A75');d.text((61,yy+34),v[1],font=f(45,True),fill='#188584')
    elif i==3:text(d,'적재량 20대 → 1대\n총 이동 5대 → 3대\n같은 관측으로 다시 계산',(61,y+55),f(31),'#526A75',545,22)
    elif scene['kind']=='assistant':text(d,'“전체 계획을 정리해줘”\n\n총대수·거리·시간과 방문 순서\n참고한 자료 확인',(61,y+55),f(31),'#526A75',535,19)
    elif scene['kind']=='close':text(d,'예측  →  계획  →  설명\n\n시범 운영으로 현장 효과 확인',(61,y+60),f(33),'#526A75',545,24)
    else:text(d,'대여소 1,294곳\n10분 뒤 재고 예측\n부족 예상과 현재 재고 확인',(61,y+55),f(32),'#526A75',545,24)
    raw=Image.open(shot).convert('RGB')
    if scene.get('crop'):raw=raw.crop(tuple(scene['crop']))
    elif scene['kind']=='assistant' and 'cards' not in scene:raw=raw.crop((370,253,830,850))
    pic=ImageOps.contain(raw,(1168,798),Image.Resampling.LANCZOS)
    xx=690+(1168-pic.width)//2;yy=118+(798-pic.height)//2
    d.rounded_rectangle((xx-5,yy-5,xx+pic.width+5,yy+pic.height+5),radius=10,fill='#DCE6EB');im.paste(pic,(xx,yy))
    ImageDraw.Draw(im).rectangle((0,934,1920,1080),fill='#173E50')
    return im

timeline=[];offset=0;segments=[];voice_metadata=[]
for i,scene in enumerate(spec['scenes']):
    duration=scene['seconds'];shot=shots/scene['shot'];base=out/f'scene-{i}.png'
    if scene['kind']!='latent':frame(i,scene,shot).save(base)
    motion=sorted(shots.glob('02-route-motion-???.jpg')) if scene['kind']=='route' else []
    for n,path in enumerate(motion):frame(i,scene,path).save(out/f'scene-{i}-frame-{n:03d}.png')
    wav=audio/f'voice-{i}.wav'
    with wave.open(str(wav)) as w:spoken=w.getnframes()/w.getframerate()
    tempo=max(1.0,spoken/(duration-0.8));voice_metadata.append(dict(scene=i,raw_seconds=spoken,atempo=tempo))
    subtitle=out/f'scene-{i}.ass';parts=[];start=0.4;total=sum(len(x) for x in scene['lines']);available=min(spoken/tempo,duration-0.8)
    def stamp(t):
        cs=round(t*100);return f'{cs//360000}:{cs//6000%60:02d}:{cs//100%60:02d}.{cs%100:02d}'
    for n,line in enumerate(scene['lines'],1):
        end=start+available*len(line)/total
        caption_draw=ImageDraw.Draw(Image.new('RGB',(1,1)))
        lines=wrap(line,caption_draw,f(39),1720)
        if len(lines)==2 and caption_draw.textlength(lines[1],font=f(39))<400:
            words=line.split()
            breaks=[(' '.join(words[:j]),' '.join(words[j:])) for j in range(1,len(words))]
            lines=min(breaks,key=lambda pair:max(caption_draw.textlength(part,font=f(39)) for part in pair))
        parts.append(f'Dialogue: 0,{stamp(start)},{stamp(end)},Default,,0,0,0,,'+'\\N'.join(lines));start=end
    header='[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 2\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Malgun Gothic,40,&H00FFFFFF,&H00FFFFFF,&H00503E17,&H00503E17,0,0,0,0,100,100,0,0,1,0,0,2,90,90,36,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
    subtitle.write_text(header+'\n'.join(parts)+'\n',encoding='utf-8')
    # Subtitle path is relative to the controlled output folder; avoid Windows colon escaping.
    filters=f"scale=1920:1080,tpad=stop_mode=clone:stop_duration={duration},ass=scene-{i}.ass:fontsdir='C\\:/Windows/Fonts'"
    clip=out/f'clip-{i}.mp4'
    if scene['kind']=='latent':
        from render_flow_frames import frames
        pattern=(out/f'scene-{i}-flow-%03d.jpg').resolve()
        fps=frames(shot,pattern,duration)
        visual=['-framerate',str(fps),'-i',str(pattern)]
    elif motion:
        visual=['-framerate','10','-i',str((out/f'scene-{i}-frame-%03d.png').resolve())]
    else:
        visual=['-loop','1','-framerate','25','-i',str(base.resolve())]
    command=[a.ffmpeg,'-hide_banner','-loglevel','error','-y',*visual,'-i',str(wav.resolve()),'-vf',filters,'-af',f'atempo={tempo:.6f},apad','-t',str(duration),'-r','25','-c:v','libx264','-threads','4','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000','-movflags','+faststart',str(clip.resolve())]
    subprocess.run(command,cwd=out,check=True);segments.append(clip)
    timeline.append(dict(start=offset,end=offset+duration,title=scene['title'],source=scene['shot']));offset+=duration
concat=out/'concat.txt';concat.write_text('\n'.join("file '"+p.name+"'" for p in segments),encoding='utf-8')
dest=out/'tashu-demo-2min.mp4';subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat.resolve()),'-c','copy','-movflags','+faststart',str(dest.resolve())],check=True)
manifest=dict(duration_seconds=offset,resolution='1920x1080',fps=25,media='Narrated walkthrough using actual browser captures and traced Flow Matching when specified; response waits edited',basis=spec['recorded_basis'],tts='Microsoft Heami Desktop, rate1',timeline=timeline,voice=voice_metadata,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size,render_location='dislab',api_training=False,field_dispatch=False)
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False))
