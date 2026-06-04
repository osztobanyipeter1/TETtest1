import numpy as np
import pandas as pd


def csv_to_ply(csv_path, ply_path):
    df = pd.read_csv(csv_path)

    required_columns = ["x", "y", "z", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Hiányzó oszlop: {col}")

    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    labels = df["label"].values

    num_points = len(df)
    unique_labels = np.unique(labels)

    np.random.seed(42)
    color_map = {}
    for label in unique_labels:
        color_map[label] = np.random.randint(0, 256, size=3, dtype=np.uint8)

    r = np.zeros(num_points, dtype=np.uint8)
    g = np.zeros(num_points, dtype=np.uint8)
    b = np.zeros(num_points, dtype=np.uint8)

    for label, color in color_map.items():
        mask = labels == label
        r[mask] = color[0]
        g[mask] = color[1]
        b[mask] = color[2]

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(num_points):
            f.write(
                f"{x[i]:.4f} {y[i]:.4f} {z[i]:.4f} {r[i]} {g[i]} {b[i]}\n"
            )


if __name__ == "__main__":
    csv_to_ply("/home/buvr_tp4/Downloads/random_test_point_cloud_(prediction).csv", "random_test_point_cloud_(prediction).ply")