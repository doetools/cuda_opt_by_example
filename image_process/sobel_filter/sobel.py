#%%
from PIL import Image
import numpy as np

#%%
img = Image.open("original.png").convert("L")
original = np.asanyarray(img)
# img.save("greyscale.png")

# %%
rows, columns = original.shape
Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
sobel_filtered_image = np.zeros(shape=(rows, columns))

for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(Gx, original[i : i + 3, j : j + 3]))  # x direction
        gy = np.sum(np.multiply(Gy, original[i : i + 3, j : j + 3]))  # y direction
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx**2 + gy**2)


# %%
sobel_filtered_image = sobel_filtered_image.astype(dtype=np.uint8)
new_img = Image.fromarray(sobel_filtered_image, "L")
new_img.show()
new_img.save("edge_py.png")

# %%
# read img data from cuda output and plot

cuda_data = np.fromfile("./edge_cuda.txt", dtype=np.int32, sep="\n")
print(cuda_data.max())
cuda_data = cuda_data.reshape(original.shape)
cuda_data = cuda_data.astype(np.uint8) + np.uint8(1)
cuda_img = Image.fromarray(cuda_data, "L")
cuda_img.show()
cuda_img.save("edge_cuda.png")


#%%
cuda = cuda_data.astype(np.int32)
py = sobel_filtered_image.astype(np.int32)

for i in cuda:
    if i == 0:
        print(i)

# diff = cuda - py

# M, N = diff.shape

# for i in range(M):
#     for j in range(N):
#         if diff[i, j] >= 55:
#             print(i, j, cuda[i, j], py[i, j])

# %%
