# Proceed with your image segmentation code
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from tensorflow.keras.preprocessing import image

# KaggleHub üzerinden DeepLabV3 modelini indir ve doğru şekilde yükle
model_path = 'C:/Users/User/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1/1.tflite'
model = tf.lite.Interpreter(model_path=model_path)  # TFLite modelini doğru şekilde kullan

def load_image(img_path):
    img = image.load_img(img_path, target_size=(257, 257))  # Resize the image to 257x257
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = img_array / 255.0  # Normalize the image to [0, 1] range
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)  # Convert to FLOAT32 tensor
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension (since the model expects batch input)
    return img_array

# Segmentasyon maskesini renklendirme (rastgele renkler kullanarak)
def decode_segmentation(mask):
    num_classes = np.max(mask) + 1  # Sınıf sayısını maskenin maksimum değeri ile belirle
    colormap = np.random.randint(0, 256, size=(num_classes, 3))  # Her sınıf için rastgele RGB renkleri oluştur
    mask_colored = colormap[mask]  # Maskeye rastgele renkleri uygula
    return mask_colored

# Segmentasyonu yapılan görüntüyü elde etme
def segment_image(image_path):
    img = load_image(image_path)
    # DeepLabV3 modelini kullanarak segmentasyonu yap
    model.allocate_tensors()  # TFLite modelini başlat
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Girdi verisini modele ver
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()  # Modeli çalıştır

    # Segmentasyon maskesini al
    segmentation_mask = model.get_tensor(output_details[0]['index'])[0]

    # Maskeyi sınıf indekslerine dönüştür ve renklendir
    decoded_mask = decode_segmentation(segmentation_mask.argmax(axis=-1))  # En olası sınıfı seç
    return img[0], decoded_mask  # Orijinal ve segment edilmiş maskeyi döndür

# Segmentasyonu ve orijinal görüntüyü görselleştir
def visualize_segmentation(image_path):
    original_image, segmented_image = segment_image(image_path)

    # Segmentasyonu ve orijinal resmi görselleştir
    plt.figure(figsize=(10, 10))

    # Orijinal resmi göster
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    # Segmentasyonu göster
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis('off')

    plt.show()

# Test görüntüsünü yükleyin ve görselleştirin
image_path = 'C:/Users/User/Desktop/YeniaySrc/WorkspaceAI/DeepLearning/TermProject/image.jpg'  # Görüntü yolu
visualize_segmentation(image_path)