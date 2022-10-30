from itertools import chain
from random import randint
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, History
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow import keras


def celsius_to_farenheit(c: int) -> float:
    return c * 1.8 + 32


def generate_data():
    return np.array([randint(-100, 100) for _ in range(20)])


def get_expected_data(input_data):
    return np.array([celsius_to_farenheit(c) for c in input_data])


def almost_equal(m, threshold=0.01):
    x, y = m
    return abs(x - y) < threshold


class TargetLossEarlyStopping(EarlyStopping):
    def __init__(self, target_val=0.00001):
        super().__init__(monitor="loss")
        self.target_val = target_val

    def on_epoch_end(self, epoch, logs={}):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if current <= self.target_val:
            print(
                f"\n\n\nReached {self.target_val} loss value so cancelling training!\n\n\n"
            )
            self.model.stop_training = True


trainingStopCallback = TargetLossEarlyStopping()

input_data = generate_data()
expected_data = get_expected_data(input_data)

model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation="linear"))
model.compile(loss="mean_squared_error", optimizer=Adam(0.1))

epochs = 2000
history: History = model.fit(
    input_data,
    expected_data,
    epochs=epochs,
    verbose=0,
    callbacks=[trainingStopCallback],
)

test_data = generate_data()
expected_test_data = get_expected_data(test_data)
actual_data = model.predict(test_data)

print(f"input data: {test_data}")
print(f"expected data: {expected_test_data}")
print(f"actaul data: {[e[0] for e in actual_data]}")
print(f"epochs: {len(history.epoch)}/{epochs}")

plt.plot(history.history["loss"])
plt.grid(True)
plt.show()

w = model.get_weights()
w = (w[0][0][0], w[1][0])
print(f"weights: {w}")

assert all(map(almost_equal, zip(w, (1.8, 32))))
