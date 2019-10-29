# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick

# Use standard LaTeX font on plots
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


def generate_plot_with_reference(x,
                                 x_ref,
                                 func,
                                 reference,
                                 quantity_name,
                                 file_name,
                                 ref_label,
                                 every_n=0,
                                 set_log_y=False):
    # We force the formatting so that all rendered images are the same size.
    # pyplot will change the size of the plot depending on how many significant
    # digits are shown...
    class ScalarFormatterForceFormat(mtick.ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

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
        ax.plot(x_ref, reference, '-', linewidth=linewidth, label=ref_label)

    ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(r"$x$", fontsize=fontsize, labelpad=-5)
    ax.set_ylabel(quantity_name, fontsize=fontsize)

    plt.grid(b=True, which='major', linestyle='--')

    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))

    ax.yaxis.set_major_formatter(yfmt)
    ax.yaxis.offsetText.set_fontsize(fontsize - 4)

    ax.xaxis.set_major_formatter(ScalarFormatterForceFormat())
    ax.xaxis.offsetText.set_fontsize(fontsize - 4)

    if quantity_name != "Local Order":
        ax.legend(loc='best', fontsize=fontsize - 1)

    if set_log_y:
        plt.yscale('log')
    plt.savefig(file_name, transparent=True, format='pdf', bbox_inches='tight')
