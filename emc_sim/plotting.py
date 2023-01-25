"""
Visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
from emc_sim import options, events


def visualizeGradientPulse(givenAx, gradientArray, pulseArray):
    # plt.style.use('ggplot')
    # fig = plt.figure(figsize=(8, 5), dpi=200)
    x = np.arange(gradientArray.shape[0])
    gradColor = '#29856c'
    pulseColor = '#5a2985'
    phaseColour = '#ff6666'
    # create axes
    pax = givenAx.twinx()
    axLimit = 1.2 * np.max(np.abs(gradientArray))
    givenAx.set_ylim(-axLimit, axLimit)
    paxLimit = np.max([1e-9, 1.25 * np.max(np.abs(pulseArray))])
    pax.set_ylim(-paxLimit, paxLimit)

    # tweaking grad
    givenAx.tick_params(axis='y', colors=gradColor)
    givenAx.set_xlabel('sampling point')
    givenAx.set_ylabel('gradient strength [mT/m]')

    # tweaking pulse
    pax.set_ylabel('pulse strength [a.u.]')
    pax.spines['left'].set_color(gradColor)
    givenAx.yaxis.label.set_color(gradColor)
    pax.spines['right'].set_color(pulseColor)
    pax.yaxis.label.set_color(pulseColor)
    pax.tick_params(axis='y', colors=pulseColor)
    pax.spines['bottom'].set_color('gray')
    pax.grid(None)

    # plot grad
    # vertical lines
    givenAx.vlines(0, 0, gradientArray[0], linewidth=3, color=gradColor)
    givenAx.vlines(x[-1], 0, gradientArray[-1], linewidth=3, color=gradColor)
    # plot and fill
    givenAx.plot(x, gradientArray, color=gradColor, linewidth=3, label='gradient')
    givenAx.fill_between(x, 0, gradientArray, color=gradColor, alpha=0.4, hatch='/')

    # plot and fill pulse
    pax.plot(x, np.abs(pulseArray), color=pulseColor, linewidth=3, label='pulse')
    pax.fill_between(x, 0, np.abs(pulseArray), color=pulseColor, alpha=0.4, hatch='/')
    phase = np.angle(pulseArray)
    mapped_phase = phase / np.pi * paxLimit / 1.25
    p_idx = np.argmax(np.abs(mapped_phase))
    pax.scatter(x, mapped_phase, color=phaseColour, s=5)
    pax.annotate(f'{phase[p_idx]/np.pi * 180.0:.1f} °', (x[p_idx], mapped_phase[p_idx]+0.05*paxLimit), color=phaseColour)

    # legend
    lines, labels = givenAx.get_legend_handles_labels()
    lines2, labels2 = pax.get_legend_handles_labels()
    pax.legend(lines + lines2, labels + labels2, loc=0)


def visualizeAllGradientPulses(gp_data: list):
    """
    Plot individual pulse - gradient profiles
    NEEDS gpList to be a List of dictionaries with specific entries from pulse-preparation module

    :param gp_data: list of g-p dictionaries per echo train
    :return: plot
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(7, 3.5 * gp_data.__len__()), dpi=200)
    for pos_idx in range(gp_data.__len__()):
        plotData = gp_data[pos_idx]
        # create axes
        ax = fig.add_subplot(gp_data.__len__(), 1, pos_idx+1)
        gpType = plotData.pulse_type
        ax.set_title(f"Pulse type: {gpType}")
        if gpType == "Acquisition":
            plotGrad = plotData.data_grad * np.ones(int(plotData.dt_sampling_steps))
            plotPulse = np.zeros(int(plotData.dt_sampling_steps))
        else:
            plotGrad = plotData.data_grad
            plotPulse = plotData.data_pulse
        visualizeGradientPulse(ax, plotGrad, plotPulse)

    plt.tight_layout()
    plt.show()


def visualizePulseProfile(tempData: options.SimulationTempData, phase=False, name=f""):
    array_mags = tempData.magnetizationPropagation
    if phase:
        cols = 2
    else:
        cols = 1
    if isinstance(array_mags, list):
        p_num = array_mags.__len__() - 1
    else:
        p_num = array_mags.shape[0] - 1
    m_fig = plt.figure(figsize=(12, 4 * p_num), dpi=200)
    m_fig.suptitle(name)
    for k_ind in np.arange(1, p_num+1):
        plot_array = array_mags[k_ind][0] + 1j * array_mags[k_ind][1]
        m_ax = m_fig.add_subplot(p_num, cols, (cols * (k_ind-1) + 1))
        m_ax.plot(tempData.sampleAxis, np.abs(plot_array),
                  color='green', linewidth=0.75)
        m_ax.fill_between(tempData.sampleAxis, np.abs(plot_array),
                          color='green', alpha=0.5)
        m_ax.set_ylabel('transverse magnetization strength', color='green')
        m_ax.tick_params(axis='y', labelcolor="green")
    if phase:
        for k_ind in np.arange(1, p_num + 1):
            plot_array = array_mags[k_ind][0] + 1j * array_mags[k_ind][1]
            m_ax = m_fig.add_subplot(p_num, cols, (cols * (k_ind - 1) + 2))
            m_ax.plot(tempData.sampleAxis,
                      np.angle(plot_array) / np.pi,
                      color='#043600', linewidth=0.75)
            m_ax.set_ylabel(r'transverse magnetization phase [$\pi$]', color='green')
            m_ax.tick_params(axis='y', labelcolor="green")
    plt.tight_layout()
    plt.show()
    return


def visualizeSequenceScheme(gp_data: list, timing: events.Timing, acquisition: events.GradPulse):
    yGrad = gp_data[0].data_grad
    yPulse = gp_data[0].data_pulse
    for idx in range(gp_data.__len__()-1):
        # concat time delay
        yGrad = np.concatenate(
            (yGrad, np.zeros(int(timing.time_pre_pulse[idx] / 5))))
        # divide by 5us to get rough no of sampling pts
        yPulse = np.concatenate(
            (yPulse, np.zeros(int(timing.time_pre_pulse[idx] / 5))))
        # pulse
        yGrad = np.concatenate((yGrad, gp_data[idx+1].data_grad))
        yPulse = np.concatenate((yPulse, gp_data[idx+1].data_pulse))
        # concat time delay
        yGrad = np.concatenate(
            (yGrad, np.zeros(int(timing.time_post_pulse[idx] / 5))))
        yPulse = np.concatenate((yPulse, np.zeros(int(timing.time_post_pulse[idx] / 5))))
        # concat acquisition
        yGrad = np.concatenate(
            (yGrad, - np.linspace(acquisition.data_grad[0], acquisition.data_grad[0],
                                  int(acquisition.duration / 5))))
        yPulse = np.concatenate((yPulse, np.zeros(int(acquisition.duration / 5))))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 4), dpi=200)
    ax = fig.add_subplot()
    visualizeGradientPulse(ax, yGrad, yPulse)
    plt.tight_layout()
    plt.show()


def visualizeEchoes(signalArray, simParams: options.SimulationParameters):
    fig = plt.figure(figsize=(10, 4), dpi=200)
    for k in range(signalArray.shape[0]):
        ax = fig.add_subplot(simParams.sequence.ETL, 3, 3 * k + 1)
        ax.plot(np.arange(50), np.real(signalArray[k, :]), label='real')
        ax.legend()
        ax = fig.add_subplot(simParams.sequence.ETL, 3, 3 * k + 2)
        ax.plot(np.arange(50), np.imag(signalArray[k, :]), label='imag')
        ax.legend()
        ax = fig.add_subplot(simParams.sequence.ETL, 3, 3 * k + 3)
        ax.plot(np.arange(50), np.abs(signalArray[k, :]), label='abs')
        ax.legend()
    plt.show()


def visualizeSignalResponse(emcCurve, t2b1: tuple = None):
    fig = plt.figure(figsize=(7, 4), dpi=200)
    ax = fig.add_subplot()
    if t2b1 is not None:
        ax.set_title(f't2: {t2b1[0]*1000:.1f}, b1: {t2b1[1]:.2f}')
    ax.set_xlabel(f'echo number')
    ax.set_ylabel(f'signal response intensity')
    ax.plot(np.arange(len(emcCurve)), emcCurve)
    plt.show()


def plotMagnetization(tempData: options.SimulationTempData):
    fig = plt.figure(figsize=(8, 8), dpi=200)

    real = tempData.magnetizationPropagation[-1][0]
    imag = tempData.magnetizationPropagation[-1][1]
    absolute = np.linalg.norm(tempData.magnetizationPropagation[-1][0:2], axis=0)
    z = tempData.magnetizationPropagation[-1][2]
    x_ax = tempData.sampleAxis * 1e3

    ax = fig.add_subplot(211)
    ax.set_xlabel(f'slice position [mm]')
    ax.set_ylabel(f'transverse magnetization [a.u.]')
    ax.plot(x_ax, real, color='#29856c', label="real")
    ax.plot(x_ax, imag, color='#5a2985', label="imag")
    ax.legend()

    ax = fig.add_subplot(212)
    ax.set_xlabel(f'slice position [mm]')
    ax.set_ylabel(f'magnetization [a.u.]')
    ax.set_ylim(-1.1, 1.1)
    ax.fill_between(x_ax, absolute, color='#29856c', alpha=0.5)
    ax.plot(x_ax, absolute, color='#29856c', label="absolute")
    ax.plot(x_ax, z, color='#5a2985', label="z")
    ax.legend()
    plt.show()
