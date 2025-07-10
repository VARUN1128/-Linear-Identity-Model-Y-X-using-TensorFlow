
# ➗ Linear Identity Model: Y = X using TensorFlow

This project demonstrates the **simplest deep learning model possible** — where the target output `Y` is exactly equal to the input `X`. It serves as a foundational example to understand how neural networks learn linear relationships, even with a single neuron and no bias!

> 🧠 Ideal for beginners in AI/ML | 🔬 Explains core ML concepts | 🚀 Built using TensorFlow

---

## 📌 Project Summary

In this project, we implement a minimal deep learning model using:

- A single training example `X = 3`, `Y = 3`
- A neural network with **1 dense neuron**, **no bias**, and **linear activation**
- Optimization using **Stochastic Gradient Descent (SGD)** and **Mean Squared Error (MSE)**
- Training for **100 epochs**
- Predicting output for new unseen input (`X = 12`)

Even with its simplicity, the model **learns the identity function Y = X** successfully.

---

## 🛠️ Tools & Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy

---

## 📋 Algorithm

1. Start  
2. Import the required libraries  
3. Create arrays `X` and `Y` with a single element each  
4. Create a `Sequential` model  
5. Add a dense layer with 1 neuron (`use_bias=False`)  
6. Compile using **SGD** optimizer and **MSE** loss  
7. Train the model for 100 epochs  
8. Predict the output for a new input (e.g., 12)  
9. Print the predicted output  
10. Stop

---

## 📄 Source Code

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Input data
x = np.array([3.0])
y = np.array([3.0])

# Step 2: Create model
model = Sequential()
model.add(Dense(1, name='D1', input_dim=1, use_bias=False))

# Step 3: Compile model
model.compile(optimizer='sgd', loss='mse')

# Step 4: Train model
model.fit(x, y, epochs=100)

# Step 5: Predict
prediction = model.predict(np.array([12]))
print(prediction)
````

---

## 📈 Training Snapshot (Loss Output)

```
Epoch 1/100 - loss: 23.6124
Epoch 2/100 - loss: 15.8770
Epoch 3/100 - loss: 10.6757
...
Epoch 100/100 - loss: 0.000002
```

✅ Final Prediction Output:

```python
[[11.999998]]
```

As expected, the model accurately learns the identity function `Y = X`.

---

## 📊 Why This Matters

Even the most complex models are built upon simple principles. This project:

* Teaches how weights are updated via gradient descent
* Demonstrates linear learning with no bias term
* Shows that deep learning can start simple

---

## 📁 Project Structure

```
📂 y-equals-x-model/
 ├── model.py
 └── README.md
```

---

## 🚀 Future Extensions

* Add bias and observe the effect
* Train with multiple data points
* Visualize the weight after training
* Use more complex functions like Y = 2X + 5 or Y = sin(X)

---

## 🌟 Star This Repo

If this helped you understand deep learning foundations better, consider giving this project a ⭐ — your support matters!

---

## 🧑‍💻 Author

**Varun Haridas**
📧 [varun.haridas321@gmail.com](mailto:varun.haridas321@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/your-profile) • [GitHub](https://github.com/your-username)

---

## 📝 License

This project is licensed under the MIT License.

```

---

