from random import randint

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, History
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def celsius_to_fahrenheit(c: int) -> float:
    return c * 1.8 + 32


def generate_data():
    return np.array([randint(-100, 100) for _ in range(10000)])


def get_expected_data(input_data):
    return np.array([celsius_to_fahrenheit(c) for c in input_data])


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
            self.model.stop_training = True


trainingStopCallback = TargetLossEarlyStopping()

input_data = generate_data()
expected_data = get_expected_data(input_data)

model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation="linear"))
model.compile(loss="mean_squared_error", optimizer=Adam(0.1))

epochs = 100
history: History = model.fit(
    input_data,
    expected_data,
    epochs=epochs,
    callbacks=[trainingStopCallback],
)

test_data = generate_data()
expected_test_data = get_expected_data(test_data)
actual_data = model.predict(test_data)

print(f"epochs: {len(history.epoch)}/{epochs}")

w = model.get_weights()
w = (w[0][0][0], w[1][0])
print(f"weights: {w}")

plt.plot(history.history["loss"])
plt.grid(True)
plt.title("Loss/Epoch")
plt.show()

assert all(map(almost_equal, zip(w, (1.8, 32))))
