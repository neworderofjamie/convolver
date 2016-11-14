import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

all_data = np.load("layer_0.npy")
num_axis = int(np.ceil(np.sqrt(all_data.shape[3])))

fig, axes = plt.subplots(num_axis, num_axis)

images = []
for x, y in itertools.product(range(num_axis), repeat=2):
    data = np.zeros((all_data.shape[1], all_data.shape[2]))
    image = axes[x,y].imshow(data, interpolation="nearest", vmin=0, vmax=1)
    images.append(image)

def update(frame):
        global all_data, num_axis, images
        for a, image in zip(range(all_data.shape[3]), images):
            image.set_array(all_data[frame,:,:,a])
        return images


anim = animation.FuncAnimation(fig, update, interval=20.0, blit=True, frames=all_data.shape[0], repeat=True)
plt.show()