import numpy as np

# Load the point cloud.
file = "data/point_clouds/tiki.pwn"
point_cloud_np = np.loadtxt(file)[:, :3]

# Write to C4D's ASCII format (for C4D Structure Manager).
with open("tmp_c4d.txt", "w") as f:
    f.write("Point  X	Y	Z\n")
    for row in range(point_cloud_np.shape[0]):
        f.write(
            f"{row}\t{point_cloud_np[row, 0]} cm\t{point_cloud_np[row, 1]} cm\t{point_cloud_np[row, 2]} cm\n"
        )
