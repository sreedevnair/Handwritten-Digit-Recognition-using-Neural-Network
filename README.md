# Recognizing Handwritted Digits using Neural Network

In this project, we will be creating a neural network that can regonize and predict handwritten digits. 

We will be using **TensorFlow** to build our neural network.

## Notes :-
### 1. Importing all the modules
For this project, we will need 3 modules :-
1. `cv2` : We are using this module to read the handwritten image and use it as the input for the model to predict the actual digit.
2. `numpy` : We are using numpy to manipulate the input image.
3. `tensorflow` : To create the neural network :)

### 2. Dataset
We will be using MNIST dataset, which is a collection of labelled handwritten digits. It has 60,000 training images and 10,000 testing image.
We can directly use the dataset through TensorFlow (Keras).

### 3. Splitting Our Dataset
The <dataset_name>.load_data() returns 2 tuple, each containing 2 arrays. First array is the *feature* dataset and the second array is the *target* dataset. Likewise, the first tuple returned is the training set (containing 60,000 training images) and the second tuple returned is the testing set (containing 10,000 testing images).

Syntax : `(X_train, y_train), (X_test, y_test) = <dataset_name>.load_data()`

### 4. Normalizing the data
Here, we are bascially trying to normalize the feature dataset into more standard values that will help the neural network to read the data more easily and efficiently.
Since our feature dataset (images in form of arrays) can hold values upto 255, we need to normalize it to the range between 0 to 1.

### 5. Creating the model (aka Neural Network)
**a.** `model = tf.keras.models.Sequential()`
<br>
This initializes a **Sequential** model which is a linear stack of layers. We can add layers to this model one by one, and the data will flow through these layers in the order they are added.
This model is good for building a feedforward neural network, where the output of one layer is the input to the next.

**b.** `model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))`
<br>
This layer doesn't contain neurons with weights and biases. Instead, it reshapes the input data. It converts a 2D matrix (like an image) into a 1D vector. 
Flatten turns this 28x28 matrix into a 784-element vector (since 28 * 28 = 784).

**c.** `model.add(tf.keras.layers.Dense(128, activation='relu'))`
<br>
A **dense layer** is a fully connected layer where each neuron receives input from all the neurons in the previous layer. It's the core layer of most neural networks. This adds a dense layer with 128 neurons to the model. 
The activation function applied to this layer is *ReLU* (Rectified Linear Unit). ReLU introduces non-linearity into the model, which helps the network learn complex patterns. It converts all negative values to zero and keeps positive values unchanged.

**d. Adding a Second Dense Layer**
<br>
Adding more layers allows the model to learn more complex representations of the data. 

**e.** `model.add(tf.keras.layers.Dense(10, activation='softmax'))`
<br>
This adds the final dense layer with 10 neurons. Since we’re working with the MNIST dataset (which has 10 digits: 0-9), we need 10 output neurons.
<br>
`activation='softmax'` converts the output values of that layer (also known as logits) into probabilities, with each neuron’s output representing the probability that the input image belongs to that class (0-9).
Softmax is commonly used in the output layer of classification networks. The neuron with the highest probability is activated and its corresponding digit is outputted as the predicted digit.

### 6. Configuring the model for training
We use `model.compile()` function to configure the model like specifying the optimizer, the loss function, and the metrics that the model should track during training and evaluation.
<br>
<br>
**a.** `optimizer='adam'`
<br>
The optimizer is an algorithm that adjusts the weights and biases of the model in order to minimize the loss function during training. Here, we are using `adam` optimzer which adjust the learning rate adaptively for each parameter in the model based on the history of gradients calculated for that parameter 🙏..
<br>
<br>
**b.** `loss='sparse_categorical_crossentropy'`
<br>
Loss fucntion measures the difference between the predicted output and the actual target value. Our goal is to minimize loss function during the model training. 
1. **sparse** : It indicates that the labels are provided as integers rather than one-hot encoded vectors .
2. **categorical_crossentropy** : It measures how well the predicted probabilities (output of the softmax layer) match the actual labels.

So, basically sparse categorical crossentropy is used to measure how far the predictions are from the actual labels, where the labels are provided as integers.

**c.** `metrics=['accuracy']`
<br>
Metrics are used to evaluate the performance of the model during training and testing. The loss function is what the optimizer seeks to minimize, metrics provide additional information about the model's performance.
**accuracy** is a common metric for classification problems. It calculates the percentage of correct predictions out of the total predictions.

### 7. Training the model
Here, we train the model using the training set (X_train, y_train). 
<br><br>
**Epochs**: An epoch is one complete pass through the entire training dataset. 
<br>
`epochs=3`: This specifies that the model should go through the entire X_train and y_train datasets 3 times, adjusting its parameters after each pass.

#### What Happens During model.fit() ?
1. **Forward Pass**: The model makes predictions on X_train based on its current parameters.
2. **Loss Calculation**: The loss function compares the model's predictions with the true labels y_train to calculate how far off the predictions are.
3. **Backward Pass (Gradient Descent)**: The optimizer (in this case, Adam) updates the model's parameters based on the gradients of the loss with respect to the parameters.
4. **Repeat**: Steps 1-3 are repeated for the number of epochs specified.

### 8. Evaluating the model
Once the model is trained, we can evalute the model using the test dataset (X_test, y_test). This will return the model's prediction loss and its accuracy.

### 9. Predicting user given input (handwritted digit image)
Now, we read an image (in this case, the image of digit 2 that I drew with my beatiful hands 🤗) using the cv2 module. Then we convert it into a 2D array that represents the pixel intensity at each *row x column*.

After that we invert the colors (turning black pixels into white and white pixels into black). We do this because in the MNIST dataset, the background was black and the pixels that represented the digits were white.

Finally, we predict the digit using `model.predict(<img_arr>)`.
