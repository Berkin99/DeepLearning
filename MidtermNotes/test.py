
import numpy as np
import matplotlib.pyplot as plt
def relu(z):
    return max(0,z)
# Yeni eğitim verisi
data = [
    (0, 1),
    (0.5, 0.5),
    (1, 0)
]

# Model parametreleri
learning_rate = 0.1
w1, b = 0.0, 0.0  # Tek ağırlık (w1) ve bias başlangıç değerleri
epochs = 100  # Maksimum eğitim dönemi sayısı
tolerance = 0.01  # Hata toleransı

# Ağırlık ve bias güncellemelerini kaydetmek için depolama
history = {
    'epoch': [],
    'w1': [],
    'b': [],
    'total_error': []
}

# Eğitim döngüsü
for epoch in range(epochs):
    total_error = 0.0
    
    for x1, y in data:
        # Ağırlıklı toplam ve ReLU aktivasyonu
        z = w1 * x1 + b
        y_pred = relu(z)
        
        # Hata hesaplama
        error = y - y_pred
        total_error += abs(error)
        
        # Ağırlık ve bias güncellemesi
        w1 += learning_rate * error * x1
        b += learning_rate * error
    
    # Verileri kaydet
    history['epoch'].append(epoch)
    history['w1'].append(w1)
    history['b'].append(b)
    history['total_error'].append(total_error)
    
    # Hata küçükse eğitim sonlandır
    if total_error < tolerance:
        break

# Ağırlık güncellemeleri gösterimi
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Ağırlık ve bias değişimi
ax[0].plot(history['epoch'], history['w1'], label='w1', marker='o')
ax[0].plot(history['epoch'], history['b'], label='bias (b)', marker='^')
ax[0].set_title('Ağırlık ve Bias Güncellemeleri')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Değer')
ax[0].legend()
ax[0].grid(True)

# Hata değişimi
ax[1].plot(history['epoch'], history['total_error'], label='Toplam Hata', color='red', marker='s')
ax[1].set_title('Toplam Hata Değişimi')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Toplam Hata')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Son model çizgisi ve verilerin gösterimi
x_vals = np.linspace(0, 1, 100)
y_vals = relu(w1 * x_vals + b)

# Eğitim verisi ve model çizgisi
plt.figure(figsize=(8, 6))
for x1, y in data:
    plt.scatter(x1, y, color='blue', label='Gerçek Veri' if x1 == 0 and y == 1 else "")

plt.plot(x_vals, y_vals, color='green', label='Perceptron Tahmin Çizgisi')
plt.title('Perceptron Tahmin Çizgisi ile Eğitim Verisi')
plt.xlabel('x1')
plt.ylabel('Model Tahmini / Gerçek Değer')
plt.legend()
plt.grid(True)
plt.show()
