from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import numpy as np

def render_scatter_som(X, W=None, W_grid=None, excitement=None, title="SOM Fitting", show=True):
    fig, ax = plt.subplots(figsize=[12.0, 8.0])
    ax.set_title(title)
    ax.scatter(X[:, 0], X[:, 1], color='blue', s=12)
    if W is not None:
        if excitement is not None:
            colors = [(1.0, 0, 1.0-e) for e in excitement]
            sizes = [20*(1+2*e) for e in excitement]
            ax.scatter(W[:, 0], W[:, 1], c=colors, s=sizes)
        else:
            ax.scatter(W[:, 0], W[:, 1], color='red', s=30)
    if W_grid:
        for pair in W_grid:
            w1, w2 = W[pair[0]], W[pair[1]]
            line = mlines.Line2D(
                (w1[0], w2[0]),
                (w1[1], w2[1]),
                color='magenta', alpha=0.5)
            ax.add_line(line)
    if show:
        plt.show()
    else:
        plt.savefig(title + ".png")

def render_som_animation(X, W_history, W_grid=None, X_history=None, title="SOM Fitting", show=True):

    W_updates_history = [W_history[i] - W_history[i-1] for i in range(1, len(W_history))]
    W_updates_history.append(np.zeros(W_history[0].shape))

    W_updates_magnitude = [(np.apply_along_axis(np.linalg.norm, 1, W)) for W in W_updates_history]
    W_updates_magnitude = [W/(0.001 + np.sum(W)) for W in W_updates_magnitude]

    fig, ax = plt.subplots(figsize=[12.0, 8.0])
    plt.title(title)
    ax.scatter(X[:, 0], X[:, 1], color='blue', s=12)

    clearlist = []

    def update(frame_number):

        print("Frame {}".format(frame_number))

        W = W_history[frame_number%len(W_history)]
        W_updates = W_updates_magnitude[frame_number%len(W_updates_magnitude)]

        while clearlist:
            c = clearlist.pop()
            c.remove()
            del c

        clearlist.append(ax.scatter(W[:, 0], W[:, 1], color='red', s=20))

        if X_history:
            x = X_history[frame_number%len(X_history)]
            clearlist.append(ax.scatter(x[0], x[1], color='green', s=26))
            for i, (w, w_upd) in enumerate(zip(W, W_updates)):
                if w_upd > 0:
                    line = mlines.Line2D(
                        (w[0], x[0]),
                        (w[1], x[1]),
                        color='green', alpha=w_upd)
                    clearlist.append(ax.add_line(line))
                    clearlist.append(ax.annotate(str(i), ( w[0] + 0.05, w[1] + 0.05 ) ))

        if W_grid:
            for pair in W_grid:
                w1, w2 = W[pair[0]], W[pair[1]]
                line = mlines.Line2D(
                    (w1[0], w2[0]),
                    (w1[1], w2[1]),
                    color='magenta', alpha=0.5)
                clearlist.append(ax.add_line(line))

        clearlist.append(ax.annotate(str(frame_number%len(W_history)), (0.02, 0.02), textcoords='axes fraction' ))

    ani = animation.FuncAnimation(fig, update, interval=200, blit=(not show), save_count=(len(W_history)-1))

    if show:
        plt.show()
    else:
        ani.save(title + '.mp4', writer='ffmpeg', fps=2, bitrate=2048)

