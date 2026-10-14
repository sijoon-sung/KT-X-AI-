"""Raw Tashu trip ZIP -> compact npz with bike ids, times, stations, minutes, km."""
import csv, io, re, zipfile, sys
import numpy as np

ZIP = r'C:/Users/DISLAB/Downloads/대전교통공사_대전시 공영자전거 타슈 대여이력 정보_20260331.zip'
OUT = sys.argv[1]
DT_RE = re.compile(r'(\d{4}-\d\d-\d\d)[ T](\d{1,2}):(\d\d)(?::(\d\d))?')


def norm(v):
    if len(v) == 19:
        return v
    m = DT_RE.match(v)
    return f'{m[1]} {int(m[2]):02d}:{m[3]}:{m[4] or "00"}' if m else None


bike_id, st_id, st_name = {}, {}, {}
cols = {k: [] for k in ('bike', 'rt', 'rs', 'tt', 'ts', 'minutes', 'km')}
with zipfile.ZipFile(ZIP) as zf:
    for name in sorted(n for n in zf.namelist() if n.lower().endswith('.csv')):
        dn = name
        try:
            dn = name.encode('cp437').decode('cp949')
        except Exception:
            pass
        ym = re.search(r'(\d\d)년(\d\d)월', dn)
        prefix = f'20{ym[1]}-{ym[2]}' if ym else None
        buf = {k: [] for k in cols}
        with zf.open(name) as fh:
            r = csv.reader(io.TextIOWrapper(fh, encoding='cp949', errors='replace'))
            next(r)
            for x in r:
                if len(x) < 19 or not x[2] or not x[10]:
                    continue
                a, b = norm(x[1]), norm(x[9])
                if a is None or b is None:
                    continue
                if prefix and not a.startswith(prefix):
                    continue
                for sid, nm in ((x[2], x[3]), (x[10], x[11])):
                    if sid not in st_id:
                        st_id[sid] = len(st_id)
                    if sid not in st_name or '\ufffd' in st_name[sid]:
                        st_name[sid] = nm
                buf['bike'].append(bike_id.setdefault(x[0], len(bike_id)))
                buf['rt'].append(a); buf['rs'].append(st_id[x[2]])
                buf['tt'].append(b); buf['ts'].append(st_id[x[10]])
                try:
                    buf['minutes'].append(float(x[17]))
                except ValueError:
                    buf['minutes'].append(np.nan)
                try:
                    buf['km'].append(float(x[18]))
                except ValueError:
                    buf['km'].append(np.nan)
        print(dn, prefix, len(buf['bike']), flush=True)
        cols['bike'].append(np.array(buf['bike'], np.int32))
        cols['rs'].append(np.array(buf['rs'], np.int32))
        cols['ts'].append(np.array(buf['ts'], np.int32))
        cols['rt'].append(np.array(buf['rt'], 'datetime64[s]').astype(np.int64))
        cols['tt'].append(np.array(buf['tt'], 'datetime64[s]').astype(np.int64))
        cols['minutes'].append(np.array(buf['minutes'], np.float32))
        cols['km'].append(np.array(buf['km'], np.float32))

d = {k: np.concatenate(v) for k, v in cols.items()}
bikes = np.array(sorted(bike_id, key=bike_id.get))
stations = np.array(sorted(st_id, key=st_id.get))
names = np.array([st_name.get(s, '') for s in stations])
np.savez(OUT, **d, bikes=bikes, stations=stations, st_names=names)
print('rows', len(d['bike']), 'bikes', len(bikes), 'stations', len(stations))
