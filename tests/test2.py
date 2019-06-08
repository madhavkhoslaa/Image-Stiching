from Operation import Stitch
import skimage
import matplotlib.pyplot as plt
instance= Stitch()
result= instance.sticher(["1.jpg","2.jpg","3.jpg"])
fig, ax = plt.subplots(figsize=(15, 12))

ax.imshow(result, cmap='gray')
fig.savefig("memememe.jpeg")
plt.tight_layout()
ax.axis('off');