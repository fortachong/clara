import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Numpy version
def xyz_numpy(frame, idxs, topx, topy, cx, cy, fx, fy):
    u = idxs[:,1]
    v = idxs[:,0]
    z = frame[v,u]
    x = ((u + topx - cx)*z)/fx
    y = ((v + topy - cy)*z)/fy
    return x, y, z


# Returns point cloud
def transform_xyz(
        depth_frame, 
        topx, bottomx, 
        topy, bottomy, 
        depth_threshold_min, depth_threshold_max,
        cx, cy, fx, fy
    ):
    point_cloud = None
    if depth_frame is not None:
        dframe = depth_frame.copy()
        # Limit the region
        dframe = dframe[topy:bottomy+1, topx:bottomx+1]
        # filter z
        filter_cond_z = (dframe > depth_threshold_max) | (dframe < depth_threshold_min)
        # ids of the filtered dframe
        dm_frame_filtered_idxs = np.argwhere(~filter_cond_z)
        point_cloud = xyz_numpy(
            dframe, 
            dm_frame_filtered_idxs,
            topx,
            topy,
            cx,
            cy,
            fx,
            fy
        )
        return point_cloud


# Matplotlib plot Initialization for Right hand xz plot
def init_plot_xz(
        width, height,
        x, z,
        centroid_x,
        centroid_z,
        min_z, max_z, 
        antenna_x, antenna_z, 
        min_x_min_z, max_x_min_z, 
        min_x_max_z, max_x_max_z
    ):
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(width*px, height*px))
    # set limits
    ax.set_xlim((antenna_x - 150, max_x_min_z + 150))
    ax.set_ylim((min_z - 100, max_z + 100))
    # antenna location
    ax.plot(antenna_x, antenna_z, marker='X', color='b', ms=8)
    ax.text(antenna_x + 5, antenna_z - 8, 'ANTENNA', color='b', fontsize='large')
    # draw limiting region
    x0, x1 = ax.get_xlim()
    ax.axline((x0, min_z), (x1, min_z), ls='dashed', color='r', linewidth=1.2)
    ax.axline((x0, max_z), (x1, max_z), ls='dashed', color='r', linewidth=1.2)
    ax.axline((antenna_x, min_z), (min_x_max_z, max_z), ls='dashed', color='r', linewidth=1.2)
    ax.axline((max_x_min_z, min_z), (max_x_max_z, max_z), ls='dashed', color='r', linewidth=1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    # Random initial plot (will be update every frame)
    plot = ax.scatter(x, z, marker='+', color='k', s=2.2)
    centroid_plot = ax.scatter(centroid_x, centroid_z, marker='X', color='r', s=20)
    # draw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot

# Plot xz (Right Hand)
def plot_xz(
        x,
        z,
        centroid_x,
        centroid_z,
        fig,
        ax,
        plot,
        centroid_plot
    ):
    plot.set_offsets(np.stack([x,z], axis=1))
    centroid_plot.set_offsets(np.array([centroid_x,centroid_z]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot


# Matplotlib plot Initialization for Left hand yz plot
def init_plot_yz(
        width, height,
        y, z,
        centroid_y,
        centroid_z,
        min_z, max_z, 
        min_y_min_z, max_y_min_z, 
        min_y_max_z, max_y_max_z
    ):
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(width*px, height*px))
    # set limits
    ymin = min(min_y_min_z, max_y_min_z, min_y_max_z, max_y_max_z)
    ymax = max(min_y_min_z, max_y_min_z, min_y_max_z, max_y_max_z)
    ax.set_ylim((ymin - 150, ymax + 150))
    ax.set_xlim((min_z - 100, max_z + 100))
    # draw limiting region
    y0, y1 = ax.get_ylim()
    ax.axline((min_z, y0), (min_z, y1), ls='dashed', color='r', linewidth=1.2)
    ax.axline((max_z, y0), (max_z, y1), ls='dashed', color='r', linewidth=1.2)
    ax.axline((min_z, min_y_min_z), (max_z, min_y_max_z), ls='dashed', color='r', linewidth=1.2)
    ax.axline((min_z, max_y_min_z), (max_z, max_y_max_z), ls='dashed', color='r', linewidth=1.2)
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    # Random initial plot (will be update every frame)
    plot = ax.scatter(z, y, marker='+', color='k', s=2.2)
    centroid_plot = ax.scatter(centroid_z, centroid_y, marker='X', color='r', s=20)
    # draw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot


# Plot yz (Left Hand)
def plot_yz(
        y,
        z,
        centroid_y,
        centroid_z,
        fig,
        ax,
        plot,
        centroid_plot
    ):
    plot.set_offsets(np.stack([z, y], axis=1))
    centroid_plot.set_offsets(np.array([centroid_z, centroid_y]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot