import seaborn as sns
import pandas as pd


def line_plot(results_path: str, x: str, y: str, algorithms: list = None, **kwargs):
    """ Creates a line plot based on metrics in a pickle file

    Args:
        results_path (str): Path to the file containing the results
        x (str): Metric on the x-axis
        y (str): Metric on the y-axis
        algorithms (list): Algorithms on which the results should be filtered

    Returns:
        figure
    """
    results = pd.read_pickle(results_path)
    results.loc[results['trial_id'] == 'af6b3_00000', 'algorithm'] = 'MPS-TD3 (fail)'

    if algorithms is not None:
        results = results[results['algorithm'].isin(algorithms)]

    if x == 'algorithm' or algorithms is not None and len(algorithms) == 1:
        plot = sns.lineplot(data=results, x=x, y=y, errorbar='sd', **kwargs)
    else:
        plot = sns.lineplot(data=results, x=x, y=y, errorbar='sd', hue='algorithm', **kwargs)

    sns.despine()
    return plot


def bar_chart(results_path: str, x: str, y: str, algorithms: list = None,
              include_not_solved: bool = True, **kwargs):
    """ Creates a bar chart based on metrics in a pickle file

    Args:
        results_path (str): Path to the file containing the results
        x (str): Metric on the x-axis
        y (str): Metric on the y-axis
        algorithms (list): Algorithms on which the results should be filtered
        include_not_solved (bool): Whether to filter unsolved episodes out

    Returns:
        figure
    """
    results = pd.read_pickle(results_path)

    if algorithms is not None:
        results = results[results['algorithm'] in list]

    if not include_not_solved:
        results = results[results['solved'] == True]

    if y == 'fraction_solved':
        results = results.groupby([x, 'algorithm'] if x != 'algorithm' else [x])['solved'].agg(sum='sum', count='count')
        results = results['sum'] / results['count']
        results = results.reset_index()
        results.columns = [x, 'algorithm', y] if x != 'algorithm' else [x, y]

    if x == 'algorithm' or algorithms is not None and len(algorithms) == 1:
        ax = sns.barplot(data=results, x=x, y=y, **kwargs)
    else:
        ax = sns.barplot(data=results, x=x, y=y, hue='algorithm', **kwargs)

    return ax


def box_plot(results_path: str, x: str, y: str, algorithms: list = None, **kwargs):
    """ Creates a box plot based on metrics in a pickle file

    Args:
        results_path (str): Path to the file containing the results
        x (str): Metric on the x-axis
        y (str): Metric on the y-axis
        algorithms (list): Algorithms on which the results should be filtered

    Returns:
        figure
    """
    results = pd.read_pickle(results_path)

    if algorithms is not None:
        results = results[results['algorithm'] in list]

    if x == 'algorithm' or algorithms is not None and len(algorithms) == 1:
        plot = sns.boxplot(data=results, x=x, y=y, **kwargs)
    else:
        plot = sns.boxplot(data=results, x=x, y=y, hue='algorithm', **kwargs)

    return plot


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """ Calculates figure size for the latex document

    Args:
        width_pt (float): Width of the latex document
        fraction (float): Fraction of the width to use for the plot
        subplots (tuple): Matrix of subplots

    Returns:
        width (float)
        height (float
    """
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** .5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in
