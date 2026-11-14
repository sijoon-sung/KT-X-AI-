"""Fetch one bounded OpenStreetMap extract that fits all comparison routes."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser=argparse.ArgumentParser();parser.add_argument('--plans',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--case',action='append',help='Route IDs to cover in the map')
args=parser.parse_args()
cases=json.loads(args.plans.read_bytes())['cases']
keys=set(args.case or ('taepyeong-2050','return-8-2050','split-2050'))
points=[xy for case in cases if case['id'] in keys for xy in case['geometry']['coordinates']]
lon,lat=zip(*points)
bounds=[round(min(lon)-.003,5),round(min(lat)-.003,5),round(max(lon)+.003,5),round(max(lat)+.003,5)]
url='https://api.openstreetmap.org/api/0.6/map.json?bbox='+','.join(map(str,bounds))
if args.out.is_file() and json.loads(args.out.read_bytes()).get('source')==url:raise SystemExit(0)
with urlopen(Request(url,headers={'User-Agent':'TashuMapStory/1.0 (+https://github.com/sijoon-sung/KT-X-AI-)'}),timeout=35) as response:
    raw=response.read(20_000_001)
if len(raw)>20_000_000:raise ValueError('Map exceeded limit')
args.out.write_text(json.dumps(dict(bounds=bounds,map=json.loads(raw),source=url),separators=(',',':')),encoding='utf-8')
