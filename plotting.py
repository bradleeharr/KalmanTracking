import matplotlib.pyplot as pl
import numpy as np


def plot_kalman_filter_results(
    filtered_state_means,
    smoothed_state_means,
    measurements,
    max_number_frames,
    model="velocity",
    obj_id="1",
):
    pl.figure(figsize=(12, 7))
    x = np.linspace(1, max_number_frames, max_number_frames)
    pl.subplot(2, 2, 1)
    x_obs_scatter = pl.scatter(x, measurements.T[0], marker="x", color="b", label="observations")
    x_position_line = pl.plot(
        x,
        filtered_state_means[:, 0],
        linestyle="-",
        marker=".",
        color="r",
        label="position est.",
        alpha=0.4,
    )
    x_smoothed_line = pl.plot(
        x,
        smoothed_state_means[:, 0],
        linestyle="-",
        marker=".",
        color="g",
        label="smoothed position est.",
        alpha=0.2,
    )
    pl.legend(loc="lower right")
    pl.title(f"Kalman Filtered X Position using {model} Model - Object {obj_id}")
    pl.xlim(xmin=0, xmax=x.max())
    pl.ylabel("X Distance")

    pl.subplot(2, 2, 2)
    y_obs_scatter = pl.scatter(x, measurements.T[1], marker="x", color="b", label="observations")
    y_position_line = pl.plot(
        x,
        filtered_state_means[:, 1],
        linestyle="-",
        marker=".",
        color="r",
        label="position est.",
        alpha=0.4,
    )

    y_smoothed_line = pl.plot(
        x,
        smoothed_state_means[:, 1],
        linestyle="-",
        marker=".",
        color="g",
        label="smoothed position est.",
        alpha=0.2,
    )
    pl.legend(loc="lower right")
    pl.title(f"Kalman Filtered Y Position using {model} Model - Object {obj_id}")
    pl.xlim(xmin=0, xmax=x.max())
    pl.ylabel("Y Distance")

    pl.subplot(2, 2, 3)
    x_velocity_line = pl.plot(
        x,
        filtered_state_means[:, 2],
        linestyle="-",
        marker=".",
        color="r",
        label="filter velocity est.",
        alpha=0.4,
    )
    x_velocity_line = pl.plot(
        x,
        smoothed_state_means[:, 2],
        linestyle="-",
        marker=".",
        color="g",
        label="smoothed velocity est.",
        alpha=0.2,
    )
    pl.xlabel("Frame Number")
    pl.legend(loc="lower right")
    pl.title(f"Modeled X Velocity using {model} Model - Object {obj_id}")

    pl.subplot(2, 2, 4)
    y_velocity_line = pl.plot(
        x,
        filtered_state_means[:, 3],
        linestyle="-",
        marker=".",
        color="r",
        label="velocity est.",
        alpha=0.4,
    )
    y_velocity_line = pl.plot(
        x,
        smoothed_state_means[:, 3],
        linestyle="-",
        marker=".",
        color="g",
        label="smoothed velocity est.",
        alpha=0.2,
    )
    pl.xlabel("Frame Number")
    pl.legend(loc="lower right")
    pl.title(f"Modeled Y Velocity using {model} Model - Object {obj_id}")

    pl.show()
