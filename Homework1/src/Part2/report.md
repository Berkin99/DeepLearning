

### Part 2 Report: 2D Object Recognition using CNNs

Report:

The generation of images containing eight classes requires the introduction of a specific type of noise. The size and diversity of the dataset play a significant role in the training of the model. 2D Shape Generator https://github.com/TimoFlesch/2D-Shape-Generator repo used for dataset generation.

In the initial experiment, a total of 800 images were generated, with 100 images per class. The number of nodes in the AlexNet architecture was kept as is. This approach resulted in both time inefficiencies and excessive fluctuation during model training. It was observed that architectures with a large number of nodes exhibit unstable behavior when trained on small datasets.

Subsequently, the dataset was expanded to include 500 images per class. During this expansion, more attention was paid to the randomness of the shapes within the images. In the code that generates random shapes, Gaussian distribution was employed instead of uniform distribution to ensure that the expected shape appearance occurred more frequently. In the initial dataset, almost all of the images exhibited a high degree of randomness, deviating significantly from the expected shape. As the dataset consisted of .png files, iterative adjustments were made to fine-tune the randomness of the images, bringing them closer to a balanced state. The term "balanced" refers to the equilibrium between the frequency of expected shapes and the occurrence of random noisy shapes.

Modelin eğitiminde