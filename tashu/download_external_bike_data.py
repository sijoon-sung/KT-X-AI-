import urllib.request
from pathlib import Path

urls = {
    'capital_bikeshare_hourly.csv': 'https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch4/bike-sharing-dataset/hour-fixed.csv',
    'london_bike_sample.csv': 'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Ensemble-Learning-with-Python/master/Chapter02/bike_sharing.csv'
}

dest_dir = Path('data/external')
dest_dir.mkdir(parents=True, exist_ok=True)

for name, url in urls.items():
    dest_path = dest_dir / name
    print(f"Downloading {name} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {name} ({dest_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
