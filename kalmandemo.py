import numpy as np
from manim import *

class KalmanFilter(Scene):
    def construct(self):
        # Initial state
        x = np.array([[50.0], [50.0], [3.0], [2.0]])

        # State transition matrix
        dt = 1
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Process noise covariance matrix
        Q = np.eye(4)

        # Initial state estimate
        x_hat = np.array([[48.0], [49.0], [0.0], [0.0]])

        # Initial state estimate covariance matrix
        P = np.eye(4) * 100

        # Observation matrix
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        # Observation noise covariance matrix
        R = np.eye(2) * 10

        # Kalman gain
        K = np.zeros((4, 2))

        # Number of time steps
        n_steps = 50

        # Initialize the object and the estimated position dots
        obj_pos = np.array([x[0, 0], x[1, 0]])
        est_pos = np.array([x_hat[0, 0], x_hat[1, 0]])

        obj_dot = Dot(point=obj_pos, color=YELLOW).scale(1.5)
        est_dot = Dot(point=est_pos, color=BLUE)

        self.add(obj_dot, est_dot)
        self.wait(1)

        for _ in range(n_steps):
            # Move the object and add process noise
            x = A @ x + np.random.normal(0, 1, (4, 1))

            # Generate noisy observation
            z = H @ x + np.random.normal(0, np.sqrt(10), (2, 1))

            # Kalman filter prediction step
            x_hat = A @ x_hat
            P = A @ P @ A.T + Q

            # Kalman filter update step
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            x_hat = x_hat + K @ (z - H @ x_hat)
            P = (np.eye(4) - K @ H) @ P
            # Update the object and estimated position dots
            obj_new_pos = np.array([x[0, 0], x[1, 0]])
            est_new_pos = np.array([x_hat[0, 0], x_hat[1, 0]])

            obj_new_dot = Dot(point=obj_new_pos, color=YELLOW).scale(1.5)
            est_new_dot = Dot(point=est_new_pos, color=BLUE)

            # Animate the movement of the object and estimated position dots
            self.play(
                Transform(obj_dot, obj_new_dot),
                Transform(est_dot, est_new_dot),
                run_time=0.5
            )
            self.wait(0.2)

        self.wait(2)


if __name__ == "__main__":
    from manim import config

    config.media_width = "60%"
    config.frame_rate = 15
    scene = KalmanFilter()
    scene.render()