from pathlib import Path
import zipfile,csv,io,json,hashlib,collections,re
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
p=Path(a.zip)
z=zipfile.ZipFile(p);member=next(x for x in z.infolist() if x.filename.lower().endswith('.csv'));b=z.read(member)
rows=list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))));keys=list(rows[0]);n=len(rows)
def counts(index):
 c=collections.Counter((r.get(keys[index]) or '').strip() for r in rows)
 # Structured survey choices only; contact and open-ended fields never leave the server.
 if any(re.search(r'\d{2,4}[- ]?\d{3,4}[- ]?\d{4}',v) for v in c):raise ValueError('Unexpected personal contact in categorical choice')
 return [{'choice':k or '(미응답)','count':v,'percent_of_all':round(v/n*100,1)} for k,v in c.items()]
stats={str(i):counts(i) for i in [1,2,3]}
out={'title':'타슈 이용 불편 탐색 설문','responses':n,'zip_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'csv_sha256':hashlib.sha256(b).hexdigest(),'closed_questions':stats,'question_texts':{str(i):keys[i] for i in [1,2,3]},'open_question_nonblank':{'parking_followup':sum(bool((r.get(keys[4]) or '').strip()) for r in rows),'time_place':sum(bool((r.get(keys[5]) or '').strip()) for r in rows)},'privacy':{'contact_column_excluded':True,'timestamps_excluded':True,'individual_free_text_not_exported':True},'limitations':['제공된 응답 10건의 탐색 집계; 모집 방식·대표성 미확인','신원 중복 제거를 수행하지 않아 10명의 독립 응답이라고 확정하지 않음','미응답을 없음 또는 경험 없음으로 채우지 않음','선호 기능·지불 의사·운영기관 도입 의향을 묻지 않은 설문','개별 자유 응답은 집계 결과로 내보내지 않음'],'execution':'dislab','analysis_kind':'descriptive presentation support,not a model selection or new confirmatory hypothesis'}
out['headline']={'weekly_or_daily_count':sum(v['count'] for v in stats['1'] if v['choice'] in ('주 1,2회','주 3,4회','거의 매일')),'weekly_or_daily_denominator':n,'parking_difficulty_count':sum(v['count'] for v in stats['3'] if v['choice']=='예'),'parking_denominator':n}
assert all(sum(v['count'] for v in stats[str(i)])==n for i in (1,2,3))
dest=Path(a.out);dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
