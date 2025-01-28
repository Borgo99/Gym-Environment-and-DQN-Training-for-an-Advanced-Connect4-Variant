import numpy as np
from .DQN import DQN
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class AdvantageMeanLayer(layers.Layer):
  """
  Custom Keras layer for calculating the mean of the advantage values.
  This is used to normalize the advantage stream in the Dueling DQN architecture.
  """
  def call(self, inputs):
      return tf.reduce_mean(inputs, axis=1, keepdims=True)

class DuelingDQN(DQN):
  """
  A Dueling Deep Q-Network (Dueling DQN) class for training and evaluating an agent in a "Four-in-a-Row Evolution" environment.
  The Dueling DQN separates the state-value function and the action-advantage function to improve learning efficiency.
  """

  def __init__(self, state_space, action_space, env, gamma=0.95):
    """
    Initialize the Dueling DQN agent.
    Args:
    - state_space: The dimensions of the state space (e.g., the game board).
    - action_space: The number of possible actions.
    - env: The environment in which the agent operates.
    - gamma: Discount factor for future rewards.
    """
    super().__init__(state_space, action_space, env, gamma)
  
  def load_model(self, model_path):
    """
    Load a pre-trained Dueling DQN model from a file.
    Custom objects (e.g., `AdvantageMeanLayer`) are used to load the model.
    Args:
    - model_path: Path to the saved model file.
    """
    custom_objects = {'AdvantageMeanLayer': AdvantageMeanLayer}
    self.model = keras.models.load_model(
      model_path, 
      safe_mode=False,
      custom_objects=custom_objects)
    self.target_model = keras.models.clone_model(self.model)
    self.target_model.set_weights(self.model.get_weights())
    self.best_model = keras.models.clone_model(self.model)
    self.best_model.set_weights(self.model.get_weights())

  def create_model(self, 
                   shared_cnn_layers=[(32, (3, 3)), (64, (3, 3))],
                   shared_dense_layers=[128,128],
                   value_stream_layers=[128],
                   advantage_stream_layers=[128],
                   padding="same",
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss="mse",
                   metrics=["accuracy"]):
    """
    Create and compile the Dueling DQN model.
    The model architecture includes shared layers, a value stream, and an advantage stream.
    Args:
    - shared_cnn_layers: List of tuples specifying the convolutional layers in the shared network.
    - shared_dense_layers: List of integers specifying the dense layers in the shared network.
    - value_stream_layers: List of integers specifying the dense layers in the value stream.
    - advantage_stream_layers: List of integers specifying the dense layers in the advantage stream.
    - padding: Padding type for the convolutional layers.
    - optimizer: Optimizer for training the model.
    - loss: Loss function for training.
    - metrics: Metrics to monitor during training.
    Returns:
    - Compiled model: A Keras model object.
    """
    
    inputs = layers.Input(shape=(*self.state_space, 1), name='state_input')

    x = inputs
    for i, (filters, kernel_size) in enumerate(shared_cnn_layers):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding=padding,
            name=f'shared_conv_{i}')(x)
    
    x = layers.Flatten(name='flatten')(x)
    for i,layer_neurons in enumerate(shared_dense_layers):
      x = layers.Dense(
        layer_neurons, 
        activation="relu", 
        name=f"shared_dense_{i}")(x)
    
    for i,layer_neurons in enumerate(value_stream_layers):
      value_fc = layers.Dense(
        layer_neurons, 
        activation="relu", 
        name=f"value_dense_{i}")(x if i==0 else value_fc)
    value = layers.Dense(1, activation='linear', name='value')(value_fc)

    for i,layer_neurons in enumerate(advantage_stream_layers):
      advantage_fc = layers.Dense(
        layer_neurons, 
        activation="relu", 
        name=f"advantage_dense_{i}")(x if i==0 else advantage_fc)
    advantage = layers.Dense(self.action_space, activation='linear', name='advantage')(advantage_fc)
    
    # Combine Value and Advantage into Q-values
    # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    advantage_mean = AdvantageMeanLayer(
      name="advantage_mean")(advantage)
    advantage_centered = layers.Subtract(
      name='advantage_centered')([advantage, advantage_mean])
    q_values = layers.Add(name='q_values')([value, advantage_centered])
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=q_values, name='DuelingDQN')
    
    return super().create_model(model, optimizer, loss, metrics)