"""
Example code outlining the key features of the GaussianStateTools.py module
Ensure GuassianStateTools.py is in the same working directory as this file. 

Created by Ryan P. Marchildon, 2017/10/06
Last updated on 2018/07/09.

"""
import GaussianStateTools as gs
import numpy as np

# ===============================
# === BASIC ONE-MODE EXAMPLES ===
# ===============================
# Initialize and plot a one-mode Gaussian state (begins in vacuum).
psi_1 = gs.GaussianState(n_modes=1)  # initialize as instance of class GaussianState()
wigner_1 = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-6, range_max=6, range_num_steps=100)
wigner_1.plot()

# Squeeze, displace, and rotate this state. Plot after each step.
psi_1.single_mode_squeeze(mode_id=1, beta_mag=0.7, beta_phase=0)  # SQUEEZE. Note that beta_mag = r.
wigner_1A = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-6, range_max=6, range_num_steps=100)
wigner_1A.plot()

psi_1.displace(mode_id=1, alpha_mag=3, alpha_phase=np.pi/4)  # DISPLACE. Specify coherent amplitude and phase.
wigner_1B = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-6, range_max=6, range_num_steps=100)
wigner_1B.plot()

psi_1.rotate(mode_id=1, phase=np.pi/4)  # ROTATE. In this case by 45 degrees.
wigner_1C = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-6, range_max=6, range_num_steps=100)
wigner_1C.plot()

# Generate a second state and compare its overlap and fidelity with the first.
psi_2 = gs.GaussianState(n_modes=1)
psi_2.single_mode_squeeze(mode_id=1, beta_mag=1, beta_phase=0)
psi_2.displace(mode_id=1, alpha_mag=2.7, alpha_phase=np.pi/4)
psi_2.rotate(mode_id=1, phase=np.pi/4)
wigner_2 = gs.OneModeGaussianWignerFunction(state=psi_2, range_min=-6, range_max=6, range_num_steps=100)
print('The Wigner Function Overlap Between States 1 and 2 is:')
print(gs.WignerOverlap(wigner_1C, wigner_2))
print('The Ulhmann Fidelity Between States 1 and 2 is:')
print(gs.OneModeFidelity(psi_1, psi_2))
print('The Wigner Function Overlap of State 1 with Itself is:')
print(gs.WignerOverlap(wigner_1C, wigner_1C))
print('The Ulhmann Fidelity of State 1 with Itself is:')
print(gs.OneModeFidelity(psi_1, psi_1))
print('\n')

# Visualize the overlap
wigner_sum = wigner_1C  # initialize such that we share the same x_mesh and p_mesh
wigner_sum.values = (wigner_1C.values + wigner_2.values)  # reassign values, adding both wigner functions together
wigner_sum.plot()


# ===============================
# === BASIC TWO-MODE EXAMPLES ===
# ===============================
# Generate and plot a two-mode squeezed state.
# Note, that computing Wigner functions becomes very computationally expensive, so we need to keep the
# sampling low.
psi_3 = gs.GaussianState(n_modes=2)
psi_3.two_mode_squeeze(mode_1_id=1, mode_2_id=2, beta_mag=1.0, beta_phase=0.0)
wigner_3 = gs.TwoModeGaussianWignerFunction(state=psi_3, range_min=-3, range_max=3, range_num_steps=30)
wigner_3.plot_all()  # see Figure 3(b) in "Squeezed Light" review article by Lvovsky.

# Now we will show interconversion between a two-mode squeezed state and two single-mode squeezed states
psi_3.two_mode_mix(mode_1_id=1, mode_2_id=2, R0=0.5)  # mix both modes through a 50:50 splitter
wigner_3_after = gs.TwoModeGaussianWignerFunction(state=psi_3, range_min=-3, range_max=3, range_num_steps=30)
wigner_3_after.plot_all()

# Compute two-mode overlaps and Uhlmann Fidelities
print('Wigner Function Overlap, Before vs After Beamsplitter, is: %.3f' % gs.WignerOverlap(wigner_3, wigner_3_after))
print('Check: wigner_3 overlap with itself is: %.3f' % gs.WignerOverlap(wigner_3, wigner_3))
print('\n')
# Note: until we implement a clone method in our class, we need to define a new instance of the class
# for this comparison (if we write psi_3_before = psi_3, before the beamsplitter, and then transform psi_3,
# psi_3_before will transform too! Hence we need to make it a separate instance of the GaussianState class.
# Later we could implement a method such as psi_3_before = psi_3.clone, once a clone method is written.
psi_3_before = gs.GaussianState(n_modes=2)
psi_3_before.two_mode_squeeze(mode_1_id=1, mode_2_id=2, beta_mag=1.0, beta_phase=0.0)
print('Uhlmann Fidelity of States, Before vs After Beamsplitter, is %.3f' % gs.TwoModeFidelity(psi_3, psi_3_before))
print('Check: Uhlmann Fidelity of psi_3 with itself is: %.3f' % gs.TwoModeFidelity(psi_3, psi_3))
print('\n')
# Note: in general, for N-mode states, Uhlmann fidelities are better than overlaps, because there is no
# need to compute the wigner function. I have included the wigner function overlap as validation.


# ==========================
# === DISSIPATION / LOSS ===
# ==========================

" SINGLE-MODE STATE "

# Initialize, squeeze, and displace a one-mode state.
psi_1 = gs.GaussianState(n_modes=1)
psi_1.single_mode_squeeze(mode_id=1, beta_mag=0.5, beta_phase=0)  # SQUEEZE. Note that beta_mag = r.
psi_1.displace(mode_id=1, alpha_mag=1.5, alpha_phase=np.pi/4)  # DISPLACE. Specify coherent amplitude and phase.

wigner = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-4, range_max=4, range_num_steps=100)
wigner.plot()
print(psi_1.sigma)

# Apply dissipation to the one-mode state.
psi_1.dissipate(mode_id=1, transmission=0.33)
wigner = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-4, range_max=4, range_num_steps=100)
wigner.plot()
print(psi_1.sigma)

# Check: 100% transmission
psi_1 = gs.GaussianState(n_modes=1)
psi_1.single_mode_squeeze(mode_id=1, beta_mag=0.5, beta_phase=0)  # SQUEEZE. Note that beta_mag = r.
psi_1.displace(mode_id=1, alpha_mag=1.5, alpha_phase=np.pi/4)  # DISPLACE. Specify coherent amplitude and phase.
psi_1.dissipate(mode_id=1, transmission=1.0)
wigner = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-4, range_max=4, range_num_steps=100)
wigner.plot()
print(psi_1.sigma)

# Check: 0% transmission
psi_1 = gs.GaussianState(n_modes=1)
psi_1.single_mode_squeeze(mode_id=1, beta_mag=0.5, beta_phase=0)  # SQUEEZE. Note that beta_mag = r.
psi_1.displace(mode_id=1, alpha_mag=1.5, alpha_phase=np.pi/4)  # DISPLACE. Specify coherent amplitude and phase.
psi_1.dissipate(mode_id=1, transmission=0.0)
wigner = gs.OneModeGaussianWignerFunction(state=psi_1, range_min=-4, range_max=4, range_num_steps=100)
wigner.plot()
print(psi_1.sigma)


" TWO SINGLE-MODE SQUEEZED STATES "

# Initialize
psi_2 = gs.GaussianState(n_modes=2)
psi_2.single_mode_squeeze(mode_id=1, beta_mag=0.5, beta_phase=0)
psi_2.single_mode_squeeze(mode_id=2, beta_mag=0.5, beta_phase=0)
psi_2.displace(mode_id=1, alpha_mag=1.5, alpha_phase=np.pi/4)
psi_2.displace(mode_id=2, alpha_mag=1.5, alpha_phase=np.pi/4)

wigner_2A = gs.TwoModeGaussianWignerFunction(state=psi_2, range_min=-4, range_max=4, range_num_steps=40)
wigner_2A.plot_all()
print(psi_2.sigma)

# Dissipate only one of these modes
psi_2.dissipate(mode_id=2, transmission=0.33)
wigner_2B = gs.TwoModeGaussianWignerFunction(state=psi_2, range_min=-4, range_max=4, range_num_steps=40)
wigner_2B.plot_all()
print(psi_2.sigma)


" TWO-MODE SQUEEZED STATE "
# Initialize
psi_3 = gs.GaussianState(n_modes=2)
psi_3.two_mode_squeeze(mode_1_id=1, mode_2_id=2, beta_mag=0.5, beta_phase=0.0)
psi_3.displace(mode_id=1, alpha_mag=1.5, alpha_phase=np.pi/4)
psi_3.displace(mode_id=2, alpha_mag=1.5, alpha_phase=np.pi/4)

wigner_3A = gs.TwoModeGaussianWignerFunction(state=psi_3, range_min=-4, range_max=4, range_num_steps=40)
wigner_3A.plot_all()  # see Figure 3(b) in "Squeezed Light" review article by Lvovsky.
print(psi_3.sigma)

# Dissipate one of these modes
psi_3.dissipate(mode_id=2, transmission=0.33)
wigner_3B = gs.TwoModeGaussianWignerFunction(state=psi_3, range_min=-4, range_max=4, range_num_steps=40)
wigner_3B.plot_all()
print(psi_3.sigma)

print('END OF EXAMPLE.PY SCRIPT')
