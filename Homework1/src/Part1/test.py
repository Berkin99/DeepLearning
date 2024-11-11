import numpy as np
import tensorflow as tf
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

# Eğitim ve test veri kümeleri oluştur
np.random.seed(0)
X = np.random.uniform(-1, 1, (500, 8))  # 500 örnekten oluşan rastgele 8 özellik
y = np.array([function(x) for x in X])  # Her x için hedef y'yi hesapla

# Eğitim ve test verilerini ayır
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 16 ile 256 arasında çok daha sık test düğüm sayıları
nodes_per_layer = list(range(16, 257, 16))  # 16, 32, 48, ..., 256
train_errors, test_errors = [], []

# 3 gizli katmanla farklı düğüm sayılarındaki performansı gözlemle
for nodes in nodes_per_layer:
    # Modeli oluştur
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(nodes, activation="relu", input_shape=(8,)))
    model.add(tf.keras.layers.Dense(nodes, activation="relu"))
    model.add(tf.keras.layers.Dense(nodes, activation="relu"))
    
    # Çıkış katmanını ekle
    model.add(tf.keras.layers.Dense(6))  # Çıkış sayısı, y'nin eleman sayısına göre
    
    # Modeli derle
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    
    # Modeli eğit
    model.fit(X_train, y_train, epochs=150, verbose=0)
    
    # Eğitim hatasını hesapla
    y_train_pred = model.predict(X_train)
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    train_errors.append(train_mse)
    
    # Test hatasını hesapla
    y_test_pred = model.predict(X_test)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    test_errors.append(test_mse)

# Grafik çiz
plt.figure(figsize=(10, 6))
plt.plot(nodes_per_layer, train_errors, 'o-', label="Eğitim Hatası", color="blue")
plt.plot(nodes_per_layer, test_errors, 's-', label="Test Hatası", color="red")
plt.xlabel("Gizli Katmandaki Düğüm Sayısı")
plt.ylabel("Hata (MSE)")
plt.legend()
plt.title("Bias–Variance Tradeoff")
plt.grid(True)
plt.show()
