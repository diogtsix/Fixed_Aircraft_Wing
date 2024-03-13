import numpy as np

def eigenAnalysis(M, C, K, F, eigs, vec, step, tend):
    W = eigs[1]  
    t = np.arange(0, tend + step, step)
    
    # Filter eigenvalues and vectors within specified bounds
    KatoOrio = W / 3
    PanoOrio = 3 * W
    EIGS = eigs[(eigs > KatoOrio) & (eigs < PanoOrio)]
    PHI = vec[:, (eigs > KatoOrio) & (eigs < PanoOrio)]
    
    # Transform matrices
    Z = PHI.T @ C @ PHI
    Lamda = PHI.T @ K @ PHI
    Force = PHI.T @ F
    I = PHI.T @ M @ PHI
    # Ensure symmetry
    I = 0.5 * (I + I.T)
    Lamda = 0.5 * (Lamda + Lamda.T)
    Z = 0.5 * (Z + Z.T)
    
    # Initial conditions
    v0 = np.zeros([eigs.shape[0], 1])
    dv0 = v0
    taf0 = PHI.T @ M @ v0
    dtaf0 = PHI.T @ M @ dv0
    
    # Damping ratios
    Zhta = Z.diagonal() / (2 * EIGS)
    
    # Damped natural frequencies
    OmegaD = EIGS * np.sqrt(1 - Zhta**2)
    
    # Compute response for each time step
    Response = np.zeros((len(eigs), len(t)))
    for i, time in enumerate(t):
        # System response due to external force
        Fs = np.concatenate([np.zeros(Force.shape), -Force])
        Mhtroo = np.block([[Lamda - (W**2) * I, W * Z], [W * Z, (W**2) * I - Lamda]])
        v = np.linalg.solve(Mhtroo, Fs)
        taf_c = v[:len(EIGS)]
        taf_s = v[len(EIGS):]
        TAFpartial = taf_c * np.cos(W * time) + taf_s * np.sin(W * time)
        
        # Homogeneous solution
        TAFHomogeneous = np.zeros_like(TAFpartial)
        for u, zhta in enumerate(Zhta):
            ekthetis = -zhta * EIGS[u] * time
            protos = -taf_c[u] * np.cos(OmegaD[u] * time)
            klasma = (-zhta * EIGS[u] * taf_c[u] - W * taf_s[u]) / OmegaD[u]
            deuteros = klasma * np.sin(OmegaD[u] * time)
            TAF = np.exp(ekthetis) * (protos + deuteros)
            TAFHomogeneous[u] = TAF
        
        # Total response
        taf = TAFHomogeneous + TAFpartial
        V = PHI @ taf
        Response[:, i] = V

    return t, Response