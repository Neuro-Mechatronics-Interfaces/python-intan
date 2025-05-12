import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import eigh

def CFICA2(emg_data, param=None, Mu=None):
    """
    Python version of CFICA2: Kernel and Correlation Constrained FastICA for HD-EMG
    """

    # Default parameter initialization
    if param is None:
        param = {}
    param.setdefault('delay', 10)
    param.setdefault('fs', 2048)
    param.setdefault('wavelength', param['fs'] // 20)
    param.setdefault('peakinterval', param['fs'] // 40)
    param.setdefault('convergethreshold', 1e-6)
    param.setdefault('valleymode', 0)

    # Initialize Mu structure if not provided
    if Mu is None:
        Mu = {'Wave': [], 'MUpulse': [], 'S': []}

    # Whitening and delay embedding
    Xt, _, _ = whitening_pca(emg_data, param['delay'])

    if Mu['MUpulse']:
        YT = delay_embed(Mu['MUpulse'], param['delay'])
        C = YT @ Xt.T
        MUnum = Mu['S'].shape[0]
    else:
        C = None
        YT = []
        MUnum = 0

    total_num = Xt.shape[0] // (2 * param['delay'] + 1)

    while MUnum < total_num:
        Y, _ = kernel_fastica(Xt, C, param['convergethreshold'])

        threshold, St = threshold_cov(Y, param['peakinterval'])

        # For now, auto-accept threshold (interactive GUI option later)
        Y, _ = corr_constrained_ica(Xt, St, param['convergethreshold'])

        YT.append(delay_embed(Y, param['delay']))
        C = np.vstack(YT) @ Xt.T if YT else None
        Mu['MUpulse'].append(Y)
        Mu['S'].append(St)
        MUnum += 1

    # Discard redundant units
    Mu['S'], Mu['MUpulse'] = discard_redundant_spikes(Mu['S'], Mu['MUpulse'])

    # Extract waveforms
    Mu['Wave'] = waveform_estimation(emg_data, Mu['S'], param['wavelength'])

    return np.array(Mu['S']), Mu

def whitening_pca(x, delay):
    n_channels, n_samples = x.shape
    xt = []
    for i in range(n_channels):
        row = np.array([np.roll(x[i], -d) for d in range(delay + 1)])
        xt.append(row[:, :n_samples - delay])
    xt = np.vstack(xt)
    xt -= np.mean(xt, axis=1, keepdims=True)

    cov = np.cov(xt)
    eigvals, eigvecs = eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    D_inv = np.diag(1. / np.sqrt(eigvals + 1e-10))
    whitening_matrix = D_inv @ eigvecs.T
    Xt = whitening_matrix @ xt
    return Xt, whitening_matrix, xt

def delay_embed(x, delay):
    """
    Emulates the yanchi() function from MATLAB's CFICA2.
    Returns (2*delay + 1)*channels x samples matrix
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]  # shape (1, N)

    n_channels, n_samples = x.shape
    out = []
    for i in range(n_channels):
        delayed = np.array([
            np.roll(x[i], -d) if d >= 0 else np.roll(x[i], x.shape[1] + d)
            for d in range(-delay, delay + 1)
        ])
        out.append(delayed)
    out = np.vstack(out)
    return out


def kernel_fastica(X, A=None, threshold=1e-6):
    n, m = X.shape
    w = np.random.randn(n)
    w /= np.linalg.norm(w)
    Y = w @ X
    for _ in range(1000):
        w_old = w.copy()
        gw = np.tanh(Y)
        g_w_deriv = 1 - gw**2
        w = (X @ gw.T) / m - g_w_deriv.mean() * w_old
        w /= np.linalg.norm(w)
        Y = w @ X
        if np.abs(np.dot(w, w_old)) > 1 - threshold:
            break
    return Y, w

def corr_constrained_ica(X, S, threshold=1e-6):
    r = X @ S.T
    r /= np.linalg.norm(r)
    w = r.copy()
    Y = w @ X
    for _ in range(1000):
        w_old = w.copy()
        gw = np.tanh(Y)
        g_w_deriv = 1 - gw**2
        w = 0.3 * r - (X @ gw.T) / X.shape[1] + g_w_deriv.mean() * w_old
        w /= np.linalg.norm(w)
        Y = w @ X
        if np.abs(np.dot(w, w_old)) > 1 - threshold:
            break
    return Y, w

def threshold_cov(x, peak_interval):
    thresholds = np.linspace(2, np.max(x[:20]), 200)
    best_cov = np.inf
    best_th = thresholds[0]
    best_spike = None
    for th in thresholds:
        peaks, _ = find_peaks(x, height=th, distance=peak_interval)
        if len(peaks) < 2:
            continue
        intervals = np.diff(peaks)
        cov = np.std(intervals) / np.mean(intervals)
        if cov < best_cov:
            best_cov = cov
            best_th = th
            best_spike = peaks
    spike_train = np.zeros_like(x)
    if best_spike is not None:
        spike_train[best_spike] = 1
    return best_th, spike_train

def discard_redundant_spikes(S, MUpulse, threshold=0.4):
    unique_S = []
    unique_MU = []
    while len(S):
        base = S[0]
        base_mu = MUpulse[0]
        unique_S.append(base)
        unique_MU.append(base_mu)
        corr = [np.corrcoef(base, s)[0, 1] for s in S]
        S = [s for i, s in enumerate(S) if corr[i] < threshold]
        MUpulse = [s for i, s in enumerate(MUpulse) if corr[i] < threshold]
    return unique_S, unique_MU


def waveform_estimation(y, S, wavelength):
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    muapt = []
    for s in S:
        spike_indices = np.where(s == 1)[0]
        waveforms = []
        for idx in spike_indices:
            if idx >= wavelength // 2 and idx + wavelength // 2 < y.shape[1]:
                segment = y_centered[:, idx - wavelength // 2:idx + wavelength // 2]
                waveforms.append(segment)
        if waveforms:
            muapt.append(np.mean(waveforms, axis=0))
        else:
            muapt.append(np.zeros((y.shape[0], wavelength)))
    return muapt




