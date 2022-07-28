"""
Visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
from emc_sim.options import SimulationParameters, SimulationTempData


def visualizeGradientPulse(givenAx, gradientArray, pulseArray):
    # plt.style.use('ggplot')
    # fig = plt.figure(figsize=(8, 5), dpi=200)
    x = np.arange(gradientArray.shape[0])
    gradColor = '#29856c'
    pulseColor = '#5a2985'
    # create axes
    pax = givenAx.twinx()
    axLimit = 1.2 * np.max(np.abs(gradientArray))
    givenAx.set_ylim(-axLimit, axLimit)
    paxLimit = 1.25 * np.max(np.abs(pulseArray))
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

    # legend
    lines, labels = givenAx.get_legend_handles_labels()
    lines2, labels2 = pax.get_legend_handles_labels()
    pax.legend(lines + lines2, labels + labels2, loc=0)


def visualizeAllGradientPulses(gpDict: dict):
    """
    Plot individual pulse - gradient profiles
    NEEDS gpList to be a List of dictionaries with specific entries from pulse-preparation module

    :param gpList: list of g-p dictionaries per echo train
    :return: plot
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(7, 3.5 * gpDict.__len__()), dpi=200)
    counter = 0
    for key in gpDict:
        counter += 1
        plotData = gpDict[key]
        # create axes
        ax = fig.add_subplot(gpDict.__len__(), 1, counter)
        ax.set_title('Pulse type: {}'.format(plotData['pulseType']))
        visualizeGradientPulse(ax, plotData['gradientData'], plotData['pulseData'])

    plt.tight_layout()
    plt.show()


def visualizePulseProfile(array_mags, phase=False):
    if phase:
        cols = 2
    else:
        cols = 1
    p_num = array_mags.shape[0]
    m_fig = plt.figure(figsize=(12, 4 * p_num), dpi=200)
    for k_ind in range(p_num):
        m_ax = m_fig.add_subplot(p_num, cols, (cols * k_ind + 1))
        m_ax.plot(np.arange(array_mags[k_ind].shape[0]), np.linalg.norm(array_mags[k_ind][:, 0:2], axis=1),
                  color='green', linewidth=0.75)
        m_ax.fill_between(np.arange(array_mags[k_ind].shape[0]), np.linalg.norm(array_mags[k_ind][:, 0:2], axis=1),
                          color='green', alpha=0.5)
        m_ax.set_ylabel('transverse magnetization strength', color='green')
        m_ax.tick_params(axis='y', labelcolor="green")
    if phase:
        for k_ind in range(p_num):
            m_ax = m_fig.add_subplot(p_num, cols, (cols * k_ind + 2))
            m_ax.plot(np.arange(array_mags[k_ind].shape[0]),
                      np.arctan(array_mags[k_ind][:, 1] / array_mags[k_ind][:, 0]) / np.pi,
                      color='#043600', linewidth=0.75)
            m_ax.set_ylabel(r'transverse magnetization phase [$\pi$]', color='green')
            m_ax.tick_params(axis='y', labelcolor="green")
    plt.tight_layout()
    plt.show()
    return


def visualizeSequenceScheme(gpDict: dict, timingArr: np.ndarray, simParams: SimulationParameters):
    yGrad = gpDict['excitation']['gradientData']
    yPulse = gpDict['excitation']['pulseData']
    for idx in range(simParams.sequence.ETL):
        if idx == 0:
            identifier = 'refocus_1'
        else:
            identifier = 'refocus'
        yGrad = np.concatenate(
            (yGrad, np.zeros(int(timingArr[idx, 0] / 5))))  # divide by 5us to get rough no of sampling pts
        yPulse = np.concatenate((yPulse, np.zeros(int(timingArr[idx, 0] / 5))))
        yGrad = np.concatenate((yGrad, gpDict[identifier]['gradientData']))
        yPulse = np.concatenate((yPulse, gpDict[identifier]['pulseData']))
        yGrad = np.concatenate((yGrad, np.zeros(int(timingArr[idx, 1] / 5))))
        yPulse = np.concatenate((yPulse, np.zeros(int(timingArr[idx, 1] / 5))))
        yGrad = np.concatenate((yGrad, - np.linspace(simParams.sequence.gradientAcquisition, simParams.sequence.gradientAcquisition,
                                                     int(simParams.sequence.durationAcquisition / 5))))
        yPulse = np.concatenate((yPulse, np.zeros(int(simParams.sequence.durationAcquisition / 5))))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 4), dpi=200)
    ax = fig.add_subplot()
    visualizeGradientPulse(ax, yGrad, yPulse)
    plt.tight_layout()
    plt.show()


def visualizeEchoes(signalArray, simParams: SimulationParameters):
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


def visualizeSignalResponse(emcCurve):
    fig = plt.figure(figsize=(7, 4), dpi=200)
    ax = fig.add_subplot()
    ax.set_xlabel(f'echo number')
    ax.set_ylabel(f'signal response intensity')
    ax.plot(np.arange(len(emcCurve)), emcCurve)
    plt.show()


def plotMagnetization(tempData: SimulationTempData, simData: SimulationParameters):
    fig = plt.figure(figsize=(8, 8), dpi=200)

    real = tempData.magnetizationPropagation[-1][0]
    imag = tempData.magnetizationPropagation[-1][1]
    absolute = np.linalg.norm(tempData.magnetizationPropagation[-1][0:2], axis=0)
    z = tempData.magnetizationPropagation[-1][2]
    x_ax = tempData.sampleAxis

    ax = fig.add_subplot(211)
    ax.set_xlabel(f'position')
    ax.set_ylabel(f'magnetization')
    ax.plot(x_ax, real, label="real")
    ax.plot(x_ax, imag, label="imag")
    ax.legend()

    ax = fig.add_subplot(212)
    ax.set_xlabel(f'position')
    ax.set_ylabel(f'magnetization')
    ax.fill_between(x_ax, absolute, alpha=0.5)
    ax.plot(x_ax, absolute, label="absolute")
    ax.plot(x_ax, z, label="z")
    ax.legend()

    plt.show()
