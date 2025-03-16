import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

np.random.seed(42)
# Define the SimpleNeuralNetwork class
class SimpleNeuralNetwork():
    def __init__(self, input_dim, lr=0.01):
        self.lr = lr
        self.weights = np.random.randn(input_dim, 1)
        self.bias = np.random.randn(1,)

    def forward(self, x):
        self.x = x
        return (x @ self.weights) + self.bias

    def loss(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        d_loss_d_ypred = (-2 / self.x.shape[0]) * (self.y_true - self.y_pred)
        d_loss_d_weights = self.x.T @ d_loss_d_ypred
        self.weights -= self.lr * d_loss_d_weights
        self.bias -= self.lr * np.sum(d_loss_d_ypred)

    def parameters(self):
        return self.weights, self.bias

# Streamlit UI to get user inputs
st.title('Simple Neural Network Training')

# Inputs from the user
low = st.number_input("Enter low value (1-7):", min_value=1, max_value=6, value=1)
high = st.number_input("Enter high value (1-7):", min_value=2, max_value=7, value=5)

if low >= high:
    st.error("Low value must be less than High value!")
else:
    feature_length = st.slider("Enter feature length (1-7):", min_value=1, max_value=7, value=4)
    rows = st.slider("Enter number of rows (1-7):", min_value=1, max_value=7, value=3)
    epochs = st.number_input("Enter number of epochs:", min_value=1, value=100000)
    # learning_rate = st.number_input("Enter learning rate:", min_value=0.0, max_value=1.0, value=0.01)
    learning_rate = st.text_input(label="Enter learning rate", value="0.01")
    try:
        learning_rate_int = int(learning_rate.replace(".", "", 1))
    except Exception as e:
        st.error("Please input correct learning rate... ")
    else:
        learning_rate = float(learning_rate)
        if st.button("Train Neural Network"):
            # Generate input data and true output
            x = np.random.randint(low, high, (rows, feature_length))
            y_true = np.prod(x, axis=1, keepdims=True)

            st.markdown("### Dataset")
            col_names = [f"Col_{idx+1}" for idx in range(0, feature_length)]
            col_names.append("Res")
            data = np.concatenate((x,y_true), axis=1)
            st.table(pd.DataFrame(data, columns=col_names))

            # Initialize neural network and train
            nn = SimpleNeuralNetwork(input_dim=feature_length, lr=learning_rate)

            # List to store the loss values during training
            loss_history = []

            for epoch in range(epochs):
                y_pred = nn.forward(x)
                loss = nn.loss(y_true, y_pred)
                nn.backward()

                # Store the loss at each epoch
                loss_history.append(loss)

            # Get final predictions
            final_predictions = nn.forward(x)

            # Prepare data for displaying the real and predicted output in a table
            output_df = pd.DataFrame({
                'Real Output': y_true.flatten(),
                'Predicted Output': final_predictions.flatten()
            })

            # Display results
            st.write("### Final Loss Curve")
            fig, ax = plt.subplots()
            ax.plot(loss_history)
            plt.title("Loss Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            st.pyplot(fig)
            

            st.write("### Real and Predicted Output")
            st.write(output_df)
            if (loss == float("inf")) or (loss == np.nan):
                st.warning("Gradients have exploded. Try a different learning rate.")
            else:
                st.write(f"Final Loss: {loss}")