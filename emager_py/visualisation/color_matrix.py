import numpy as np
from emager_py.streamers import EmagerStreamerInterface 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

class ColorMatrix():
    def __init__(self, streamer: EmagerStreamerInterface):
        self.streamer = streamer
    
    def calculate(self):
        baseline = np.array(self.streamer.read())
        if baseline.ndim == 1:
            baseline = baseline[:, np.newaxis]  # Convert to 2D array if it is 1D
        baseline = np.median(baseline, axis=1)
        print(f"Baseline ({baseline.shape}): \n{baseline}")

        data = np.array(self.streamer.read())
        print(f"Data1 ({data.shape}): \n{data}")
        
        # Ensure that data and baseline have compatible shapes
        if len(baseline.shape) == 1:
            baseline = baseline[:, np.newaxis]

        if data.shape[0] != baseline.shape[0]:
            raise ValueError(f"Shape mismatch: Data shape {data.shape} and Baseline shape {baseline.shape} must match in the first dimension.")

        nb_pts = data.shape[1]  # Number of points in one read
        data = np.transpose(data)
        print(f"Transposed Data ({data.shape}): \n{data}")

        # Tiling baseline to match the shape of the transposed data
        baseline_tiled = np.tile(baseline.T, (nb_pts, 1))
        print(f"Tiled Baseline ({baseline_tiled.shape}): \n{baseline_tiled}")

        data = data - baseline_tiled
        print(f"Data2 ({data.shape}): \n{data}")

        # Subtract the mean of each row (across columns) from the data
        mean_data_per_row = np.mean(data, axis=1)
        data = data - mean_data_per_row[:, np.newaxis]
        print(f"Data3 ({data.shape}): \n{data}")

         # Reshape data to (nb_pts, 4, 16)
        data = np.reshape(data, (nb_pts, 4, 16))
        print(f"Data4 ({data.shape}): \n{data}")

        # Calculate mean of absolute values across the first axis
        mean_data = np.mean(np.abs(data), axis=0)
        print(f"Mean ({mean_data.shape}): \n{mean_data}")

        min = 10  # -32769
        max = 700  # 32769
        rescaled_data = (mean_data - min) / (max - min)
        print(f"Rescaled Data: {rescaled_data} // {rescaled_data.shape}")
        colors = cm.viridis(rescaled_data)
        print(f"Colors: {colors} // {colors.shape}")
        return colors
    
    def updatefig(self, *args):
        self.im.set_array(self.calculate())
        return self.im,

    def run(self):
        self.fig = plt.figure()
        self.im = plt.imshow(self.calculate(), animated=True, cmap=cm.viridis) #plt.scatter
        self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=50, blit=True)
        plt.title('Muscle activation')
        plt.show()

if __name__ == '__main__':
    from emager_py.streamers import SerialStreamer
    from emager_py.utils.find_usb import find_psoc

    PORT = find_psoc()
    stream_client = SerialStreamer(PORT, 1500000)
    color_matrix = ColorMatrix(stream_client)
    color_matrix.run()