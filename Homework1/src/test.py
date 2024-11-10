import numpy as np
import matplotlib.pyplot as plt

# Eğitim ve doğrulama verisi oluşturma fonksiyonları

# Version 1: Eğitim ve doğrulama verisini aynı veri kümesinden bölme
def version_1():
    x_data = np.random.uniform(-1, 1, 200)  # 200 örnek, tek özellikli
    y_data = np.sin(x_data)  # Örnek fonksiyon (tek özellikli)
    # Veriyi bölme (ilk 100 eğitim, geri kalan doğrulama)
    x_train, y_train = x_data[:100], y_data[:100]
    x_val, y_val = x_data[100:], y_data[100:]
    return x_train, x_val, y_train, y_val

# Version 2: Eğitim ve doğrulama verisini bağımsız iki kümeden oluşturma
def version_2():
    x_train = np.random.uniform(-1, 1, 100)  # Eğitim verisi
    y_train = np.sin(x_train)  # Eğitim hedefleri
    x_val = np.random.uniform(-1, 1, 100)  # Bağımsız doğrulama verisi
    y_val = np.sin(x_val)  # Doğrulama hedefleri
    return x_train, x_val, y_train, y_val

# Verileri al
x_train1, x_val1, y_train1, y_val1 = version_1()
x_train2, x_val2, y_train2, y_val2 = version_2()

noise1 = np.random.normal(0, 0.1, y_train1.shape)
y_train1 = y_train1 + noise1

noise2 = np.random.normal(0, 0.1, y_train2.shape)
y_train2 = y_train2 + noise2

# Grafik üzerinde görselleştirme
plt.figure(figsize=(12, 6))

# Version 1: Eğitim ve doğrulama verilerini aynı kümeden
plt.subplot(1, 2, 1)
plt.scatter(x_train1, y_train1, color='blue', label='Eğitim Verisi', alpha=0.6)
plt.scatter(x_val1, y_val1, color='red', label='Doğrulama Verisi', alpha=0.6)
plt.title("Version 1: Eğitim ve Doğrulama Verisi (Aynı Küme)")
plt.xlabel('X (Tek Özellik)')
plt.ylabel('Y')
plt.legend()

# Version 2: Eğitim ve doğrulama verilerini bağımsız kümeler
plt.subplot(1, 2, 2)
plt.scatter(x_train2, y_train2, color='blue', label='Eğitim Verisi', alpha=0.6)
plt.scatter(x_val2, y_val2, color='red', label='Doğrulama Verisi', alpha=0.6)
plt.title("Version 2: Eğitim ve Doğrulama Verisi (Bağımsız Küme)")
plt.xlabel('X (Tek Özellik)')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()
