import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from emager_py.streamers import EmagerStreamerInterface 


class RealTimeMatrixPlot:
    def __init__(self, streamer:EmagerStreamerInterface, matrix_shape=(4, 16), interval=200, colormap='viridis', min=500, max=15000):
        self.streamer = streamer
        self.matrix_shape = matrix_shape
        self.interval = interval
        self.colormap = colormap
        self.min = min
        self.max = max
        self.matrix = np.zeros(matrix_shape)
        self.matrix = self.get_matrix()

        self.fig, self.ax = plt.subplots()
        self.cax = self.ax.matshow(self.matrix, cmap=self.colormap, animated=True, vmin=0, vmax=1)
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        self.fig.colorbar(self.cax, cax=cax)

        self.ax.set_xticks(np.arange(matrix_shape[1]))
        self.ax.set_yticks(np.arange(matrix_shape[0]))
        # self.ax.grid(which='major', color='black', linestyle='-', linewidth=1)
        self.ax.set_title('Real-Time Updating Matrix of 64 Channels')

        self.ani = animation.FuncAnimation(self.fig, self.update_matrix, interval=interval, save_count=50)

    def get_matrix(self):

        # Read data from streamer
        matrix = self.streamer.read()

        if matrix is None or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            print("No data available")
            return self.matrix

        # Absolute matrix
        matrix = np.absolute(matrix)
        print(f"Matrix shape: {matrix.shape} - Min: {np.min(matrix)} - Max: {np.max(matrix)}")

        # Normalize the matrix
        min_val = self.min
        max_val = self.max
        epsilon = 1e-8  # Small epsilon value to avoid division by zero
        matrix = (matrix - min_val) / (max_val - min_val + epsilon)

        # Get the mean of each chanels
        mean = np.mean(matrix, axis=0)

        # Reshape the mean to a matrix
        matrix = np.reshape(mean, self.matrix_shape)

        return matrix

    def update_matrix(self, *args):
        self.matrix = self.get_matrix()
        self.cax.set_array(self.matrix)
        return self.cax,

    def show(self):
        plt.show()

# Usage
if __name__ == "__main__":
    from emager_py.streamers import SerialStreamer
    from emager_py.utils.find_usb import find_psoc

    PORT = find_psoc()
    stream_client = SerialStreamer(PORT, 1500000)
    rt_plot = RealTimeMatrixPlot(stream_client, colormap='hot')
    rt_plot.show()
