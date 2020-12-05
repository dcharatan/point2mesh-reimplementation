from collections import deque
from typing import Deque


class ConvergenceDetector:
    window_size: int
    threshold: float
    loss_window: Deque[float]
    running_average_window: Deque[float]

    def __init__(self, window_size: int = 25, threshold: float = 0.99) -> None:
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.loss_window = deque()
        self.running_average_window = deque()

    def step(self, loss_value: float) -> bool:
        # Update the loss window.
        self.loss_window.append(loss_value)
        if len(self.loss_window) > self.window_size:
            self.loss_window.popleft()

        # Calculate the average over the window.
        loss_sum = 0
        for loss in self.loss_window:
            loss_sum += loss
        loss_average = loss_sum / len(self.loss_window)

        # Update the running average window.
        self.running_average_window.append(loss_average)
        if len(self.running_average_window) > self.window_size:
            past_average = self.running_average_window.popleft()

            # Indicate that convergence has been detected if the running average
            # of the loss hasn't been decreasing.
            if loss_average > past_average * self.threshold:
                return True

        return False