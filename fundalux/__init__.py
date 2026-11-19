"""Fundalux: explainable referable diabetic-retinopathy screening.

The package is shared by the training notebook and the local app so that
training and inference run exactly the same preprocessing.
"""

__version__ = "0.1.0"

# International Clinical DR scale, as used by APTOS / EyePACS / IDRiD / Messidor-2.
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# Referable DR = moderate NPDR or worse. (Diabetic macular edema is also
# referable clinically, but the training sets carry no DME labels.)
REFERABLE_MIN_GRADE = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
