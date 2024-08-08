__all__ = ["display_multiple_gathers",
           "display_sidebyside",
]

import numpy as np
import matplotlib.pyplot as plt


def display_multiple_gathers(data, ishots=None, irecs=None, 
                             srcs=None, recs=None, t=None, 
                             figsize=(15, 5), vlims=None, vclip=1.,
                             cmap='gray', interpolation=None, 
                             titles=None):
    """Display multiple gathers

    Display multiple shot or receiver gathers side by side.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `ns x nt x nr`
    ishots : :obj:`tuple`, optional
        Indices of shots to display 
        (set to `None` to display receiver gathers)
    irecs : :obj:`tuple`, optional
        Indices of receivers to display
    srcs : :obj:`numpy.ndarray`, optional
        Source axis of size `ns`
    recs : :obj:`numpy.ndarray`, optional
        Receiver axis of size `nr`
    t : :obj:`numpy.ndarray`, optional
        Time axis of size `nt`
    figsize : :obj:`tuple`, optional
        Figure size
    vlims : :obj:`tuple`, optional
        Colorbar limits
    vclip : :obj:`tuple`, optional
        Colorbar clipping to be applied on top of `vlims`
    cmap : :obj:`str`, optional
        Colormap
    interpolation : :obj:`str`, optional
        Interpolation in plotting
    titles : :obj:`tuple`, optional
        Titles to use for the different panels
    
    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        Figure handle
    axs : :obj:`list`
        Axes handles

    """
    # Choose whether to display shot of receiver gathers
    xlabel = 'Recs [m]'
    titleprefix = 'Shot'
    if ishots is None:
        data = data.transpose(2, 1, 0)
        recs = srcs
        ishots = irecs
        xlabel = 'Shots [m]'
        titleprefix = 'Rec'

    # Define axes
    recs = (0, data.shape[1]) if recs is None else recs
    t = (0, data.shape[2]) if t is None else t

    # Define vlims
    if vlims is None:
        vlims = (-np.abs(data.ravel()).max(), np.abs(data.ravel()).max())

    # Display
    fig, axs = plt.subplots(1, len(ishots), sharey=True, figsize=figsize)
    for ax, ishot in zip(axs, ishots):
        ax.imshow(data[ishot], cmap=cmap,
                  vmin=vclip * vlims[0], vmax=vclip * vlims[1], 
                  extent=(recs[0], recs[-1], t[-1], t[0]),
                  interpolation=interpolation)
        ax.axis('tight')
        ax.set_xlabel(xlabel)
        ax.set_title(f'{titleprefix} {ishot}' if titles is None else titles[ishot])
    axs[0].set_ylabel('Time [s]')
    fig.tight_layout()
    return fig, axs


def display_sidebyside(data, data1, ishot, srcs, recs, t=None,
                       figsize=(15, 5), vlims=None, vclip=1.,
                       cmap='gray', titles=None):
    """Display multiple gathers

    Display multiple shot or receiver gathers side by side.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `ns x nt x nr`
    data : :obj:`numpy.ndarray`
        Second data of size `ns x nt x nr`
    ishot : :obj:`int`
        Index of shot to display
    srcs : :obj:`numpy.ndarray`
        Source array
    recs : :obj:`numpy.ndarray`
        Receiver array
    t : :obj:`numpy.ndarray`, optional
        Time axis of size `nt`
    figsize : :obj:`tuple`, optional
        Figure size
    vlims : :obj:`tuple`, optional
        Colorbar limits
    vclip : :obj:`tuple`, optional
        Colorbar clipping to be applied on top of  `vlims`
    cmap : :obj:`str`, optional
        Colormap
    titles : :obj:`tuple`, optional
        Titles to use for `data` and `data1`
    
    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        Figure handle
    axs : :obj:`list`
        Axes handles

    """
    # Extract shot gathers
    data = data[ishot]
    data1 = data1[ishot]

    # Define data to display
    izerooff = np.argmin(np.abs(srcs[ishot] - recs))
    if izerooff > len(recs) // 2:
        data = data[:, ::-1]
        data1 = data1[:, ::-1]
        recs = recs[::-1]
        izerooff = len(recs) - izerooff

    # Define axes
    off = np.abs(srcs[ishot] - recs)
    t = (0, data.shape[2]) if t is None else t

    # Define vlims
    if vlims is None:
        vlims = (-np.abs(data.ravel()).max(), np.abs(data.ravel()).max())
    
    # Define titles
    titles = (None, None) if titles is None else titles

    # Display
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw=dict(wspace=0))
    axs[0].imshow(data[:, izerooff:][:, ::-1], cmap=cmap, vmin=vclip * vlims[0], vmax=vclip * vlims[1], 
                  extent=(off[-1], off[izerooff], t[-1], t[0]))
    axs[1].imshow(data1[:, izerooff:], cmap=cmap, vmin=vclip * vlims[0], vmax=vclip * vlims[1], 
                  extent=(off[izerooff], off[-1], t[-1], t[0]))
    axs[2].imshow(data[:, izerooff:][:, ::-1], cmap=cmap, vmin=vclip * vlims[0], vmax=vclip * vlims[1], 
                  extent=(off[-1], off[izerooff], t[-1], t[0]))
    axs[0].set_title(titles[0])
    axs[1].set_title(titles[1])
    axs[2].set_title(titles[0])
    for ax in axs:
        ax.axis('tight')
        ax.set_xlabel('Offset [m]')
    axs[0].set_ylabel('Time [s]')
    return fig, ax
