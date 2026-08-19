import urllib.request
from pathlib import Path

def download_sample_video():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_file = data_dir / "sample_traffic_north.mp4"

    if sample_file.exists() and sample_file.stat().st_size > 10000:
        print(f"Sample traffic video already exists at {sample_file}")
        return sample_file

    url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
    print(f"Downloading sample traffic video from {url}...")
    try:
        urllib.request.urlretrieve(url, str(sample_file))
        print(f"Downloaded sample traffic video successfully ({sample_file.stat().st_size} bytes)")
        return sample_file
    except Exception as e:
        print(f"Direct download failed ({e}), trying mirror...")
        mirror_url = "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi"
        alt_file = data_dir / "sample_traffic_north.avi"
        urllib.request.urlretrieve(mirror_url, str(alt_file))
        print(f"Downloaded mirror traffic video to {alt_file}")
        return alt_file

if __name__ == "__main__":
    download_sample_video()
