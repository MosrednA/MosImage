import os

# The suite describes the CUDA profile on every machine; Apple silicon tests
# switch to the Metal profile explicitly. Set before the app reads its settings.
os.environ["MOSIMAGE_PREFERRED_DEVICE"] = "cuda"
os.environ["MOSIMAGE_HARDWARE_PROFILE"] = "rtx5070ti-16gb"
