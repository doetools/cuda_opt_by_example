#%%
from PIL import Image
import numpy as np

img = Image.open("original.png").convert("L")
original = np.asanyarray(img)
ref = np.asanyarray(Image.open("edge.png").convert("L"))
generate = np.fromfile("./new_img.txt", dtype=np.uint8)

ref.shape
original.shape

print(original[:3][:3])
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
new_img.save("greyscale_py.png")

# %%
sobel_filtered_image[1][1]
# %%
