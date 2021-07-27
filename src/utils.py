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


# normal distance
def distance_(points_x, points_z, antenna_x, antenna_z):
    centroid_x = None
    centroid_z = None
    distance = None
    if not (points_x is None or points_z is None):
        centroid_x = np.mean(points_x)
        centroid_z = np.mean(points_z)
        distance = np.sqrt((centroid_x-antenna_x)**2 + (centroid_z-antenna_z)**2)
    return centroid_x, centroid_z, distance


# filter outliers
def distance_filter_out_(points_x, points_z, antenna_x, antenna_z):
    centroid_x = None
    centroid_z = None
    distance = None
    if not (points_x is None or points_z is None):
        # quantiles
        q1x, q3x = np.quantile(points_x, [0.25, 0.75])
        q1z, q3z = np.quantile(points_z, [0.25, 0.75])
        iqrx = q3x - q1x
        iqrz = q3z - q1z
        # limits to determine outliers
        liminfx = q1x - 1.5*iqrx
        limsupx = q3x + 1.5*iqrx
        liminfz = q1z - 1.5*iqrz
        limsupz = q3z + 1.5*iqrz
        # filter outliers (only include points between liminf and limsup)
        points_xz = np.vstack([points_x, points_z])
        cond_x = (points_xz[0,:] >= liminfx) & (points_xz[0,:] <= limsupx)
        cond_z = (points_xz[1,:] >= liminfz) & (points_xz[1,:] <= limsupz)
        filterd_xz = points_xz[:,cond_x | cond_z]
        if points_xz.size > 0:
            centroid_x = np.mean(filterd_xz[0,:])
            centroid_z = np.mean(filterd_xz[1,:])
            distance = np.sqrt((centroid_x-antenna_x)**2 + (centroid_z-antenna_z)**2)
    return centroid_x, centroid_z, distance        


# filter outliers and only consider fingers
def distance_filter_fingers_(points_x, points_z, antenna_x, antenna_z):
    centroid_x = None
    centroid_z = None
    distance = None
    if (points_x.size > 0 and points_z.size > 0):
        centroid_z = np.mean(points_z)
        # quantiles
        q1x, q3x = np.quantile(points_x, [0.25, 0.75])
        q1z, q3z = np.quantile(points_z, [0.25, 0.75])
        iqrx = q3x - q1x
        iqrz = q3z - q1z
        # limits to determine outliers
        liminfx = q1x - 1.5*iqrx
        limsupx = q3x + 1.5*iqrx
        liminfz = q1z - 1.5*iqrz
        limsupz = q3z + 1.5*iqrz
        # filter outliers (only include points between liminf and limsup)
        points_xz = np.vstack([points_x, points_z])
        cond_x = (points_xz[0,:] >= liminfx) & (points_xz[0,:] <= limsupx)
        cond_z = (points_xz[1,:] >= liminfz) & (points_xz[1,:] <= limsupz)
        filterd_xz = points_xz[:,cond_x | cond_z]
        if filterd_xz.size > 0:
            # further filter the 
            cond_z_fingers = (filterd_xz[1,:] <= centroid_z)
            fingers_xz = filterd_xz[:,cond_z_fingers]
            if fingers_xz.size > 0:
                centroid_x = np.mean(fingers_xz[0,:])
                centroid_z = np.mean(fingers_xz[1,:])
                distance = np.sqrt((centroid_x-antenna_x)**2 + (centroid_z-antenna_z)**2)
    return centroid_x, centroid_z, distance


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
        centroid_f_x,
        centroid_f_z,
        centroid_fing_x,
        centroid_fing_z,
        min_z, max_z, 
        antenna_x, antenna_z, 
        min_x_min_z, max_x_min_z, 
        min_x_max_z, max_x_max_z
    ):
    c_f_x = centroid_f_x
    if centroid_f_x is None:
        c_f_x = centroid_x
    c_f_z = centroid_f_z
    if centroid_f_z is None:
        c_f_z = centroid_z

    c_fing_x = centroid_fing_x
    if centroid_fing_x is None:
        c_f_x = centroid_x
    c_fing_z = centroid_fing_z
    if centroid_fing_z is None:
        c_fing_z = centroid_z        

    # plt.rcParams['figure.facecolor'] = '#646970'
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(width*px, height*px))
    # ax.set(facecolor = "#646970")
    # ax.axis('off')
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
    ax.axline((antenna_x, antenna_z), (max_x_max_z, max_z), ls='dashed', color='k', linewidth=0.5)
    ax.axline((max_x_min_z, min_z), (max_x_max_z, max_z), ls='dashed', color='r', linewidth=1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    # Random initial plot (will be update every frame)
    plot = ax.scatter(x, z, marker='+', color='k', s=2.2)
    centroid_plot = ax.scatter(centroid_x, centroid_z, marker='X', color='r', s=20)
    centroid_f_plot = ax.scatter(c_f_x, c_f_z, marker='X', color='g', s=20)
    centroid_fing_plot = ax.scatter(c_fing_x, c_fing_z, marker='X', color='b', s=20)
    # draw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot, centroid_f_plot, centroid_fing_plot


# Plot xz (Right Hand)
def plot_xz(
        x,
        z,
        centroid_x,
        centroid_z,
        centroid_f_x,
        centroid_f_z,
        centroid_fing_x,
        centroid_fing_z,
        fig,
        ax,
        plot,
        centroid_plot,
        centroid_f_plot,
        centroid_fing_plot
    ):
    plot.set_offsets(np.stack([x,z], axis=1))
    centroid_plot.set_offsets(np.array([centroid_x,centroid_z]))
    centroid_f_plot.set_offsets(np.array([centroid_f_x,centroid_f_z]))
    centroid_fing_plot.set_offsets(np.array([centroid_fing_x,centroid_fing_z]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot, centroid_f_plot, centroid_fing_plot


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