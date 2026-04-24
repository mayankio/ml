#!/usr/bin/env python3
import csv
import io
import pathlib
import urllib.request
import zipfile


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def download_text(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def download_bytes(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fetch_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    rows = []
    for raw_line in download_text(url).splitlines():
        if not raw_line.strip():
            continue
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) == 5:
            rows.append(values)
    write_csv(DATA_DIR / "iris.csv", header, rows)


def fetch_breast_cancer():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    header = [
        "id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]
    rows = []
    for raw_line in download_text(url).splitlines():
        if not raw_line.strip():
            continue
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) == len(header):
            rows.append(values)
    write_csv(DATA_DIR / "breast_cancer_wisconsin_diagnostic.csv", header, rows)


def fetch_tiny_mnist(limit=500):
    url = "https://raw.githubusercontent.com/phoebetronic/mnist/master/mnist_test.csv.zip"
    archive_bytes = download_bytes(url)
    header = ["label"] + [f"pixel_{index}" for index in range(784)]
    rows = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open("mnist_test.csv") as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8")
            reader = csv.reader(text_stream)
            next(reader, None)
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                rows.append(row)
    write_csv(DATA_DIR / "mnist_tiny.csv", header, rows)


def main():
    fetch_iris()
    fetch_breast_cancer()
    fetch_tiny_mnist()
    print(f"Downloaded datasets into {DATA_DIR}")


if __name__ == "__main__":
    main()
