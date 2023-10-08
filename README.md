### SAM-Med2D

My own custom implementation of SAM-Med2D model from scratch using torch in order to learn how SAM model works.

SAM-Med2D uses a pre-trained SAM model with additional AdapterLayer added in the attention blocks to fine-tune the model to recognise segmentation maps from medical scans which are provided via bounding boxes or prompts via the prompt encoder.

( MORE ON MODEL ARCHITECTURE ETC )


It's still a work in progress with initial ImageEncoder model built and tests added.