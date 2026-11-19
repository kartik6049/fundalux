# MATLAB / Simulink side

The problem statement asks for MATLAB and a Simulink model. The training and the app in this
repository are in Python (training had to run on a cloud GPU), so this folder is the bridge.

**These two scripts were written without access to MATLAB and have never been run.**
Treat them as a starting point: expect to fix block or parameter names for your release
(`get_param(block, 'DialogParameters')` lists the valid names).

| File | What it is for | Needs |
|---|---|---|
| `build_screening_model.m` | Builds a SimEvents model of the screening camp, the counterpart of `fundalux/screening_sim.py` | Simulink, SimEvents |
| `import_fundalux_onnx.m` | Imports one member of the trained ensemble from its `.onnx` file in `models/`, predicts and draws Grad-CAM | Deep Learning Toolbox + the ONNX converter support package |

The Python simulation (`python -m fundalux.screening_sim`) is tested and is the reference for
what the Simulink model should reproduce. The numbers the Simulink model needs (sensitivity,
specificity, share of ungradable photographs) are in `models/metrics.json` under
`simulation_inputs`.

`import_fundalux_onnx.m` imports one network at a time; the app's decision is the average of all members,
so a single member's score is not the screening result. It expects an image that has already been cropped
and enhanced (it then resizes it to that member's input size, which is close to, but not exactly, what the
Python pipeline feeds the network):

```
python -m fundalux.cli photo.jpg --save-dir out     # writes out/enhanced.png
```
