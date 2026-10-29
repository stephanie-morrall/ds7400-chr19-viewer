"""Constructs and loads the deployed three-class residual architecture."""

from training.final_model import ResidualMLP, load_final


def make_selected(metadata):
    """Creates a separate practice model with the recorded dropout setting."""
    return ResidualMLP(metadata["dropout"])


def load_selected(folder):
    """Loads the frozen predictor after checking its recorded file hashes."""
    return load_final(folder)
