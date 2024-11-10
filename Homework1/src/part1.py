import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Polinom fonksiyonunu tanımla
def function(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    y1 = x1 * x3 + 1.2 * x1 * x5 - x6 * x7 * x8 - 2 * x1**2 * x8 + x5
    y2 = x1 * x5 * x6 - x3 * x4 - 3 * x2 * x3 + 2 * x2**2 * x4 - 2 * x7 * x8 - 1
    y3 = 2 * x3**2 - x5 * x7 - 3 * x1 * x4 * x6 - x1**2 * x2 * x4 - 1
    y4 = -x6**3 + 2 * x1 * x3 * x8 - x1 * x4 * x7 - 2 * x5**2 * x2 * x4 - x8
    y5 = x1**2 * x5 - 3 * x3 * x4 * x8 + x1 * x2 * x4 - 3 * x6 - x1**2 * x7 + 2
    y6 = x1**2 * x3 * x6 - x3 * x5 * x7 + x3 * x4 + 2.2 * x4 + x2**2 * x3 - 2.1
    return [y1, y2, y3, y4, y5, y6]

np.random.seed(0)
xdata = np.random.uniform(-1, 1, (2000, 8))
ydata = np.array([function(x) for x in xdata])

xt, yt = xdata[:1000], ydata[:1000]
xv, yv = xdata[1000:], ydata[1000:]

noise = np.random.normal(0, 0.001, yt.shape)
yt = yt + noise

# Modeli tanımla
model = models.Sequential([
    layers.Dense(264, activation='relu', input_shape=(8,)),
    layers.Dense(264, activation='relu'),
    layers.Dense(264, activation='relu'),
    layers.Dense(6)
])

# Modeli derle - SGD optimizer ile
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9), loss='mse')

# Modeli eğit
history = model.fit(
    xt, yt,
    epochs=200,
    validation_data=(xv, yv),
    verbose=1
)

# Eğitim ve doğrulama kayıplarını çizdir
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Overfitting Görseli (Gürültülü Eğitim Verisi)')
plt.show()
