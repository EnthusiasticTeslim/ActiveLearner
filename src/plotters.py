import matplotlib.pyplot as plt
import pandas as pd
import paxplot as px

def parallel_plot(result):
    '''Plot the parallel plot of the result.
    Params:
        result: The result object from the optimization.
    Returns:
        None'''
    dd = pd.DataFrame(result.X, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7'])
    dd['Strength'] = -result.F[:, 0]
    dd['Biodegradability'] = -result.F[:, 1]

    cols = dd.columns

    # Create figure
    paxfig = px.pax_parallel(n_axes=len(cols))
    paxfig.plot(dd.to_numpy())

    # Add labels
    paxfig.set_labels(cols)

    # Add colorbar
    color_col = len(cols) - 2
    paxfig.add_colorbar(ax_idx=color_col, cmap='plasma', colorbar_kwargs={'label': cols[color_col]})

    # Set limits for each axis
    lower_limit = {'X1': 0, 'X2': 0, 'X3': 0, 'X4': 0, 'X5': 0, 'X6': 0.01, 'X7': 120.00}

    upper_limit = {'X1': 1, 'X2': 1, 'X3': 1, 'X4': 1, 'X5': 1, 'X6': 0.1, 'X7': 200.00}

    for idx, col in enumerate(cols[:color_col]):
        paxfig.set_lim(
                        ax_idx = idx, 
                        bottom = lower_limit[col] - 0.1 * (upper_limit[col] - lower_limit[col]), 
                        top=upper_limit[col] + 0.1 * (upper_limit[col] - lower_limit[col])
                        )

    # set figure size
    paxfig.set_size_inches(19, 8)
    paxfig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)  # Padding

    # set labels
    labels = [
                r'$\rm X1$', r'$\rm X2$', r'$\rm X3$', r'$\rm X4$',
                r'$\rm X5$', r'$\rm X6$', r'$\rm X7$', r'$\rm Strength$', r'$\rm Biodegradability$']

    paxfig.set_labels(labels)

    plt.show()

def plot_pareto_front(result):
    '''Plot the pareto front of the result.
    Params:
        result: The result object from the optimization.
    Returns:'''
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.scatter(-result.F[:,0], -result.F[:,1], color="blue")
    ax.set_xlabel("Strength")
    ax.set_ylabel("Biodegradability")
    plt.show()