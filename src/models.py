"""Model management functions for SensorMCP server."""

import os
import torch
from typing import Any, Dict, Optional

from autodistill.detection import CaptionOntology

from state import STATE, SUPPORTED_BASE_MODELS, SUPPORTED_TARGET_MODELS


# ----- Fix yolo loading issue -----
# Save the original torch.load function
_original_torch_load = torch.load


def custom_torch_load(*args, **kwargs):
    """Override torch.load to ensure weights_only=False."""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


# Override torch.load globally
torch.load = custom_torch_load


def create_ontology(objects_list: str) -> tuple[bool, str]:
    """Create an ontology from a comma-separated list of objects.
    
    Args:
        objects_list: Comma-separated list of objects to detect
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Parse comma-separated string into list
        objects = [obj.strip() for obj in objects_list.split(",") if obj.strip()]
        
        if not objects:
            return False, "Error: Empty objects list provided."

        # Generate ontology dictionary from the list
        # Using the same object name as both prompt and class name
        # Converting spaces to underscores for class names
        ontology_dict = {obj: obj.replace(" ", "_").lower() for obj in objects}

        STATE["ontology"] = CaptionOntology(ontology_dict)

        class_list = list(ontology_dict.values())
        return True, f"Successfully defined ontology with {len(ontology_dict)} classes: {class_list}"
    except Exception as e:
        return False, f"Error defining ontology: {str(e)}"


def initialize_base_model(model_name: str) -> tuple[bool, str]:
    """Initialize a base model for labeling.
    
    Args:
        model_name: Name of the base model
        
    Returns:
        Tuple of (success, message)
    """
    model_name = model_name.lower().replace("-", "_")

    # Check ontology
    if STATE["ontology"] is None:
        return False, "Error: Please define an ontology first using define_ontology()."

    # Check if model is supported
    if model_name not in SUPPORTED_BASE_MODELS:
        return False, f"Error: Unsupported base model. Please choose from: {', '.join(SUPPORTED_BASE_MODELS)}"

    try:
        if model_name == "grounded_sam":
            from autodistill_grounded_sam import GroundedSAM
            STATE["base_model"] = GroundedSAM(ontology=STATE["ontology"])
            return True, "Successfully initialized GroundedSAM base model."
    except ImportError:
        package_name = model_name.replace("_", "-")
        return False, f"Error: Module not found. Please install autodistill-{package_name} package."
    except Exception as e:
        return False, f"Error initializing base model: {str(e)}"


def initialize_target_model(model_name: str) -> tuple[bool, str]:
    """Initialize a target model for training.
    
    Args:
        model_name: Name of the target model
        
    Returns:
        Tuple of (success, message)
    """
    if model_name not in SUPPORTED_TARGET_MODELS:
        return False, f"Error: Unsupported model. Please choose from: {', '.join(SUPPORTED_TARGET_MODELS)}"

    try:
        from autodistill_yolov8 import YOLOv8
        
        # Add the unsupported global to the safe globals list
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

        # Initialize model
        STATE["target_model"] = YOLOv8(model_name)
        return True, f"Successfully initialized YOLOv8 target model with weights {model_name}."
    except ImportError:
        return False, "Error: Module not found. Please install autodistill-yolov8 package."
    except Exception as e:
        return False, f"Error initializing target model: {str(e)}"


def train_target_model(epochs: int, device: int) -> tuple[bool, str]:
    """Train the target model on labeled data.
    
    Args:
        epochs: Number of training epochs
        device: Device to use for training
        
    Returns:
        Tuple of (success, message)
    """
    # Validate parameters
    if epochs <= 0:
        return False, "Error: Number of epochs must be positive."

    # Check for data.yaml
    data_yaml_path = os.path.join(STATE["labeled_dataset"], "data.yaml")
    if not os.path.exists(data_yaml_path):
        return False, f"Error: data.yaml not found at {data_yaml_path}"

    try:
        # Train model
        STATE["target_model"].train(data_yaml_path, epochs=epochs, device=device)

        # Set the trained model path
        runs_dir = os.path.join(os.getcwd(), "runs/detect")
        STATE["trained_model_path"] = runs_dir

        return True, f"Training completed successfully with {epochs} epochs on device {device}. Model saved at {runs_dir}"
    except Exception as e:
        return False, f"Error during training: {str(e)}" 