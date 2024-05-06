import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

# unravel append numpy
def unraveled(preds):
    return np.concatenate(preds).ravel()
# calculate r2 score
def acc(dd):
    ''' dd contains the predicted values, true values, mean losses'''
    return r2_score(unraveled(dd[1]), unraveled(dd[0]))
def rmse(dd):
    ''' dd contains the predicted values, true values, mean losses'''
    return mean_squared_error(unraveled(dd[1]), unraveled(dd[0]), squared=False)
def mae(dd):
    ''' dd contains the predicted values, true values, mean losses'''
    return mean_absolute_error(unraveled(dd[1]), unraveled(dd[0]))
def evs(dd):
    ''' dd contains the predicted values, true values, mean losses'''
    return explained_variance_score(unraveled(dd[1]), unraveled(dd[0]))

# show model crossplot
def plot_data(preds_train, preds_test, data, target_col):
    ''' preds_train and preds_test are tuples of predicted and true values
    return plot of the train and test data with the 45 degree line'''
    plt.figure(figsize=(3, 2), dpi=200)
    plt.scatter(unraveled(preds_train[1]), unraveled(preds_train[0]), label=r'$\rm train$', alpha=0.7)
    plt.scatter(unraveled(preds_test[1]), unraveled(preds_test[0]), label=r'$\rm test$', alpha=0.7)
    # plot 45 degree line, get max and min values
    plt.plot([min(data[target_col]), max(data[target_col])], [min(data[target_col]), max(data[target_col])], color='red')
    plt.xlabel(r'$\rm True \ Protein \ Ads. (mg/cm^2)$', fontsize=8)
    plt.ylabel(r'$\rm Pred \ Protein \ Ads. (mg/cm^2)$', fontsize=8)

def show_performance(preds_train, preds_test): 
    ''' preds_train and preds_test are tuples of predicted and true values
    return (1) plot of the train and test data with the 45 degree line
            (2) histogram of residuals for train and test data'''
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    axs[0].scatter(unraveled(preds_train[1]), unraveled(preds_train[0]), color='red', label=r'$\rm train$', alpha=0.7)
    axs[0].scatter(unraveled(preds_test[1]), unraveled(preds_test[0]), color='green', label=r'$\rm test$', alpha=0.7)
    # plot 45 degree line, get max and min values
    axs[0].plot([-50, 450], [-50, 450], color='black')
    axs[0].set_xlabel(r'$\rm Ground \ Truth $')
    axs[0].set_ylabel(r'$\rm Predictions $')
    # zoom in by removing values beyond 2000
    axs[0].set_xlim([-50, 450])
    axs[0].set_ylim([-50, 450])
    axs[0].legend(frameon=False)

    # plot histogram of residuals
    axs[1].hist(unraveled(preds_train[1]) - unraveled(preds_train[0]), bins=50, alpha=0.7, color='red')
    axs[1].hist(unraveled(preds_test[1]) - unraveled(preds_test[0]), bins=50, alpha=0.7, color='green')
    axs[1].set_xlabel(r'$\rm Residuals$')
    axs[1].set_ylabel(r'$\rm Frequency$')
    axs[1].legend([r'$\rm train$', r'$\rm test$'], frameon=False)

    plt.tight_layout()

def plot_heat_map(data: pd.DataFrame = None, fig_size = (10, 5), save_fig: bool = False):
    ''' Plot the heatmap of the correlation matrix of the data  '''
        
    fig, ax = plt.subplots(1, figsize=fig_size, facecolor='white')

    # Create the heatmap for the original data
    sns.heatmap(data.corr(method='pearson'), cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False)
    # Show the plot
    plt.show()


def runAL(strategy, dataset, n_round: int = 10, n_query:int = 8):
    print("Round 0")
    strategy.train()
    train = strategy.predict(dataset.get_train_data()[1])
    test = strategy.predict(dataset.get_test_data())
    print(f'Round 0 accuracy --> Train: {acc(train)} Test: {acc(test)}')

    for rd in range(1, n_round+1):
        print(f"Round {rd}")
        # get index for new query
        query_idxs = strategy.query(n_query)
        # update labels
        strategy.update(query_idxs)
        strategy.train()
        # calculate accuracy
        train = strategy.predict(dataset.get_train_data()[1])
        test = strategy.predict(dataset.get_test_data())
        # print accuracy
        print(f'accuracy --> Train: {acc(train):.3f} Test: {acc(test):.3f}')
        print(f'rmse --> Train: {rmse(train):.3f} Test: {rmse(test):.3f}')
        print(f'mae --> Train: {mae(train):.3f} Test: {mae(test):.3f}')
        print(f'evs --> Train: {evs(train):.3f} Test: {evs(test):.3f}')

    return train, test