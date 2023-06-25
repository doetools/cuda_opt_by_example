#%%
from PIL import Image

img = Image.open("original.png").convert("L")
img.save("greyscale.png")
# %%
