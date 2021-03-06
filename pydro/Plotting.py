# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np

# Use standard LaTeX font on plots
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.use('Agg')

import matplotlib.pyplot as plt


# We force the formatting so that all rendered images are the same size.
# pyplot will change the size of the plot depending on how many significant
# digits are shown...
class _ScalarFormatterForceFormat(mtick.ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


def generate_plot_with_reference(x,
                                 x_ref,
                                 func,
                                 reference_solution,
                                 quantity_name,
                                 file_name,
                                 ref_label,
                                 every_n=0,
                                 set_log_y=False):
    """
    Generate a plot of a snapshot from a 1d simulation and write it to disk.

    :param list x: The x-coordinates on which `func` is defined.
    :param list x_ref: The x-coordinates on which `reference_solution`
        is defined.
    :param list func: The function/variable to plot.
    :param list reference_solution: The reference solution to plot.
    :param str quantity_name: The label for the function/variable being
        plotted.
    :param str file_name: The name of the file to write.
    :param str ref_label: The reference label. Typically either
        'Exact' or 'Reference' depending on whether the reference solution
        is an exact analytic solution or obtained from a high-resolution
        simulation.
    :param int every_n: If non-zero only plot every nth grid point in
        `func`.
    :param bool set_log_y: If `True` then the y-axis is plotted on a log
        scale.
    """
    if every_n == 0:
        every_n = len(x) // 50
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.set_size_inches(6, 6, forward=True)
    markersize = 5
    fontsize = 20
    linewidth = 1.8

    # Use a colorblind and grey scale friendly color palette.
    ax.set_prop_cycle(
        plt.cycler(color=['#0F2080', '#F5793A', '#A95AA1', '#85C0F9']))

    ax.plot(x[0::every_n],
            func[0::every_n],
            'o',
            markersize=markersize,
            label="Numerical")
    if quantity_name != "Local Order":
        ax.plot(x_ref,
                reference_solution,
                '-',
                linewidth=linewidth,
                label=ref_label)

    ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(r"$x$", fontsize=fontsize, labelpad=-5)
    ax.set_ylabel(quantity_name, fontsize=fontsize)

    plt.grid(b=True, which='major', linestyle='--')

    yfmt = _ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))

    ax.yaxis.set_major_formatter(yfmt)
    ax.yaxis.offsetText.set_fontsize(fontsize - 4)

    ax.xaxis.set_major_formatter(_ScalarFormatterForceFormat())
    ax.xaxis.offsetText.set_fontsize(fontsize - 4)

    if quantity_name != "Local Order":
        ax.legend(loc='best', fontsize=fontsize - 1)

    if set_log_y:
        plt.yscale('log')
    plt.savefig(file_name, transparent=True, format='pdf', bbox_inches='tight')


def generate_spacetime_plot(file_name,
                            var,
                            var_name,
                            x,
                            times,
                            smoothen,
                            set_log_y,
                            vmin=None,
                            vmax=None,
                            time_max_elements=100,
                            x_max_elements=200):
    """
    Generate a spacetime plot from a 1d simulation and write it to disk.

    :param str file_name: The name of the file to write.
    :param list var: A list of the variable/function to plot at each time.
       The data must be in the same order as the `times` argument.
    :param str var_name: The name of the variable being plotted.
    :param list x: The x-coordinates on which `var` is defined. Assumed
        to be time-independent.
    :param list times: A list of the times at which data was recorded
        during the evolution.
    :param bool smoothen: If `True` applies a `gouraud` smoothening to
        the data during rendering.
    :param bool set_log_y: If `True` then the log of `var` is plotted.
    :param double vmin: If specified sets the lower limit of the range
        for plotting `var`.
    :param double vmax: If specified sets the upper limit of the range
        for plotting `var`.
    :param int time_max_elements: The maximum number of cells to plot
        in time. Increasing this increases the temporal resolution but
        makes the generated images larger and more expensive to render.
    :param int x_max_elements: The maximum number of cells to plot
        in space. Increasing this increases the temporal resolution but
        makes the generated images larger and more expensive to render.
    """
    print(
        "Generating spacetime plot. This might take a minute. Please be patient."
    )
    every_n_time = 1
    time_length = len(times)
    if len(times) > time_max_elements:
        every_n_time = len(times) // time_max_elements
        time_length = len(times[0::every_n_time])

    every_n_x = 1
    if len(x) > x_max_elements:
        every_n_x = len(x) // x_max_elements

    times2 = np.zeros([len(x[0::every_n_x]), time_length])
    for i in range(len(x[0::every_n_x])):
        times2[i, :] = np.nanmean(
            np.pad(times.astype(float),
                   (0, 0 if times.size % every_n_time == 0 else every_n_time -
                    times.size % every_n_time),
                   mode='constant',
                   constant_values=np.NaN).reshape(-1, every_n_time),
            axis=1)

    xs2 = np.zeros([len(x[0::every_n_x]), time_length])
    for i in range(len(times[0::every_n_time])):
        xs2[:, i] = np.nanmean(np.pad(
            x.astype(float),
            (0,
             0 if x.size % every_n_x == 0 else every_n_x - x.size % every_n_x),
            mode='constant',
            constant_values=np.NaN).reshape(-1, every_n_x),
                               axis=1)

    # Deal with averaging the 2d array. First average space,
    # then average time...
    var_x_avg = np.zeros([var.shape[1] // every_n_x, var.shape[0]])
    for time_index in range(var.shape[0]):
        vari_copy = var[time_index, :]
        var_x_avg[:, time_index] = np.nanmean(
            np.pad(vari_copy.astype(float),
                   (0, 0 if vari_copy.size % every_n_x == 0 else every_n_x -
                    vari_copy.size % every_n_x),
                   mode='constant',
                   constant_values=np.NaN).reshape(-1, every_n_x),
            axis=1)

    var_avg = np.zeros([var_x_avg.shape[0], time_length])
    for x_index in range(var_avg.shape[0]):
        vari_copy = var_x_avg[x_index, :]
        var_avg[x_index, :] = np.nanmean(np.pad(
            vari_copy.astype(float),
            (0, 0 if vari_copy.size % every_n_time == 0 else every_n_time -
             vari_copy.size % every_n_time),
            mode='constant',
            constant_values=np.NaN).reshape(-1, every_n_time),
                                         axis=1)

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.set_size_inches(6, 6, forward=True)
    fontsize = 20

    if set_log_y:
        var_avg = np.log10(var_avg)
        var_name = r"$\log_{10}($" + var_name + r"$)$"
        var_name = var_name.replace("$$", "")

    pcm = ax.pcolormesh(xs2,
                        times2,
                        var_avg,
                        vmax=vmax,
                        vmin=vmin,
                        shading=('gouraud' if smoothen else 'flat'),
                        edgecolors='face')

    yfmt = _ScalarFormatterForceFormat()
    cbar = fig.colorbar(pcm, ax=ax, pad=0.01, format=yfmt)
    yfmt.set_powerlimits((0, 0))

    ax.yaxis.set_major_formatter(yfmt)
    ax.yaxis.offsetText.set_fontsize(fontsize - 4)

    ax.xaxis.set_major_formatter(_ScalarFormatterForceFormat())
    ax.xaxis.offsetText.set_fontsize(fontsize - 4)

    # Move axis labels closer
    ax.set_xlabel(r"$x$", fontsize=fontsize, labelpad=-5)
    ax.set_ylabel(r"$t$", fontsize=fontsize)
    plt.title(var_name, fontsize=fontsize)
    plt.savefig(file_name, transparent=True, format='pdf', bbox_inches='tight')
