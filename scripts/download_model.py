import argparse
import os
import urllib.request


def main():
    ap = argparse.ArgumentParser(description="Download pretrained model bundle.")
    ap.add_argument("--url", required=True, help="URL до файла .pkl (например, GitHub Releases).")
    ap.add_argument("--out", default="models/xgb.pkl", help="Куда сохранить.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Downloading: {args.url}")
    urllib.request.urlretrieve(args.url, args.out)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
