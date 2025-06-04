Sign-Language-Recognition-System-using-TensorFlow-in-Python-https://drive.google.com/drive/u/0/my-drive DATA SET LINK TO DOWNLOAD
step-1
Importing Libraries-
Python, importing libraries means bringing in external modules or packages so that you can use pre-written code instead of writing everything from scratch. Libraries contain useful functions, classes, and variables that help in tasks like math operations, data analysis, machine learning, and web development.
open cmd type  -pip install tensorflow,pip install numpy,pip install pandas,pip install matplotlib and warning instal python 3.10 not latest version becauce some library doesnot support latest version
codes step 1
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
step-2
load the dataset 
The dataset consists of two CSV files: sign_mnist_train.csv and sign_mnist_test.csv. Each row in these files represents a sample image. The first column (index 0) contains the label, which is a number from 0 to 25, corresponding to letters of the alphabet in American Sign Language. The remaining 784 columns hold the pixel values for a 28x28 grayscale image, where each pixel has an intensity ranging from 0 to 255.
step-2 code
from google.colab import files
uploaded = files.upload()
import pandas as pd
df = pd.read_csv('sign_mnist_train.csv')
df.head()
step-3 
The dataset is split into two separate CSV files — one for training and the other for testing. In this step, we’ll load the data and prepare it for model training. Since the hand signs for the letters 'J' and 'Z' involve motion and are not included, we’ll exclude them from classification. The function load_data(path) is used to read and process the dataset. It reshapes the flat pixel arrays into 28x28 images and converts the labels into one-hot encoded format.
code-step-3
def load_data(path):
    df = pd.read_csv('/content/sign_mnist_train.csv')
    y = np.array([label if label < 9
                  else label-1 for label in df['label']])
    df = df.drop('label', axis=1)
    x = np.array([df.iloc[i].to_numpy().reshape((28, 28))
                  for i in range(len(df))]).astype(float)
    x = np.expand_dims(x, axis=3)
    y = pd.get_dummies(y).values

    return x, y

X_train, Y_train = load_data('/content/sign_mnist_train.csv')
X_test, Y_test = load_data('/content/sign_mnist_test.csv')
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
Now let's check the shape of the training and the testing data.
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
Step 4: Data Visualization
In this step, we’ll visualize a few sample images from the dataset to get a better understanding of how different hand signs representing various letters look. This helps in gaining insight into the data and ensures that images are correctly labeled before training the model.
class_names = list(string.ascii_lowercase[:26].replace(
    'j', '').replace('z', ''))

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].squeeze(), cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(Y_train, axis=1)[i]])
plt.tight_layout()
plt.show()
Step 5: Model Development
Starting from this step, we’ll utilize TensorFlow and its Keras API to construct a Convolutional Neural Network (CNN). Keras offers convenient tools to design, compile, and train deep learning models.

Our CNN model will be built using the Sequential API and will include the following components:

Three convolutional layers, each followed by a max pooling layer.

A flattening layer to convert the 2D feature maps into a 1D vector.

Fully connected (dense) layers, ending with an output layer.

Batch normalization layers to promote faster and more stable training.

A dropout layer to help reduce overfitting.

The final dense layer will have 24 units, representing the 24 sign language classes (excluding 'J' and 'Z').

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(24, activation='softmax')
])
model.summary()
When compiling the model, we provide three essential parameters:

Optimizer: The method used to optimize the cost function (e.g., gradient descent).
Loss Function: The function used to evaluate the model's performance.
Metrics: Metrics used to evaluate the model during training and testing.
When compiling the model, we provide three essential parameters:

Optimizer: The method used to optimize the cost function (e.g., gradient descent).
Loss Function: The function used to evaluate the model's performance.
Metrics: Metrics used to evaluate the model during training and testing.
Step 6: Model Evaluation
After training the model, we will visualize the training and validation accuracy as well as the loss for each epoch. This helps us analyze how well the model is performing.
last is output 
The model achieves 99% accuracy on the test set, which is impressive for a simple CNN model.

By using just a simple CNN model we are able to achieve an accuracy of 99% which is really great. This shows that this technology is certainly going to help us build some amazing applications which can be proved a really great tool for people with some special needs.

