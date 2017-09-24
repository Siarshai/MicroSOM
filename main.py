from assembly import generate_simple_grid_of_neurons
from microsom import RectifyingSOM, HistoryLevel
from datagen import generate_five_hills_in_ring
from visualization import render_scatter_som, render_som_animation
import numpy as np

# for reproducibility
np.random.seed(1337)

# Uncomment if you want animation
history_level = HistoryLevel.NONE
# history_level = HistoryLevel.EPOCHS
# history_level = HistoryLevel.STEPS

if history_level == HistoryLevel.NONE:
    learning_rate=0.1
    epochs=50
elif history_level == HistoryLevel.EPOCHS:
    learning_rate=0.025
    epochs=10
else:
    learning_rate=0.025
    epochs=1

X = generate_five_hills_in_ring()

W, D, grid = generate_simple_grid_of_neurons(rows=8, columns=8)
som = RectifyingSOM(W, D, learning_rate=learning_rate)
som.fit(X, epochs, with_history=history_level)
W = som.get_neuron_weights()
excitement = som.get_processed_excitement()
render_scatter_som(X, W, W_grid=grid, excitement=excitement, title="RSOM-epoch-" + str(epochs))

if history_level != HistoryLevel.NONE:
    W_history = som.get_w_history()
    X_history = som.get_x_history()
    render_som_animation(X, W_history, W_grid=grid, X_history=X_history, show=True)
