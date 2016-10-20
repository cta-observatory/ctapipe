
"""
Set of diagnostic plots relating to muons 
For generic use with all muon algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from IPython import embed

def plot_muon_efficiency(source):

    """
    Plot the muon efficiencies
    """
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    figip,axip = plt.subplots(1,1,figsize=(10,10))
    figrw,axrw = plt.subplots(1,1,figsize=(10,10))

    mu_eff = []
    impact_param = []
    ring_width = []

    for mu_evt in source:
        if mu_evt[0] is not None and mu_evt[1] is not None:
            mu_eff.append(mu_evt[1].optical_efficiency_muon)
            #impact_param.append(mu_evt[1].impact_parameter/u*m)
            #ring_width.append(mu_evt[1].ring_width/u*deg)

    ax.hist(mu_eff,20)
    plt.figure(fig.number)

    axip.hist(impact_param,20)
    plt.figure(figip.number)

    axrw.hist(ring_width,20)
    plt.figure(figrw.number)

    plt.show()
