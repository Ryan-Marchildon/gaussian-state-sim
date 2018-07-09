# gaussian-state-sim

**Gaussian State Evolution Simulator for Quantum Optics**

*Created by Ryan P. Marchildon, 2018/07*

This repository consists of two python files:

1. **GaussianStateTools.py**, a python library for gaussian quantum optics.
2. **Example.py**, a worked example demonstrating the GaussianStateTools library in action. 

Gaussian States are a class of states of light (photons) important for continuous-variable
and discrete-CV hybrid approaches to photonic quantum computing and quantum information processing.

The motivation behind this project is to create a suite of easy-to-use tools for quick simulation 
of Gaussian State evolution in Quantum Optics. While the formalism for evolving Gaussian states has
existed for nearly two decades (e.g. based on symplectic transformations), this code aims to provide 
a simple object-oriented pipeline that makes it easier to simulate state evolution 
through complex sets of transformations, such as quantum circuits. 

As a quick example, the code for generating, squeezing, displacing, rotating, and dissipating 
a single-mode Gaussian State, and then computing its Wigner Function, is merely:

```python
import GaussianStateTools as gs
import numpy as np

psi_1 = gs.GaussianState(n_modes=1)  
psi_1.single_mode_squeeze(mode_id=1, beta_mag=0.7, beta_phase=0)  
psi_1.displace(mode_id=1, alpha_mag=3, alpha_phase=np.pi/4) 
psi_1.rotate(mode_id=1, phase=np.pi/4)  
psi_1.dissipate(mode_id=1, transmission=0.33)

wigner_1 = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-4, range_max=4, range_num_steps=100)
wigner_1.plot()
```

Documentation of the library's objects, methods, and functions are found within **GaussianStateTools.py**. 
For a more thorough example of how to use the library, see **Example.py**. 

This code has been written and tested for Gaussian states up to N=3 modes. Some present limitations and
room for community contributions:
* Uhlmann fidelities are available for comparing states with up to N=2 modes. A generalized N-mode Uhlmann fidelity 
exists within the literature, but has not yet been implemented here. 
* For tracing out (i.e. deleting) modes, I've included a function that works for up to N=3 modes, but this can be
futher generalized and extended to N modes. 

I hope this code is useful to the quantum optics community. 
If you'd like to get in touch, visit me at [rpmarchildon.com](http://rpmarchildon.com/).
