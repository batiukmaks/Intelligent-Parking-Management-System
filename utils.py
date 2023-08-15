import platform
import torch
import os
import string

def get_default_device():
    # If the OS is Windows, use "cuda" if available, otherwise fallback to "cpu"
    if platform.system() == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # If the system is running on a Mac with M1 processor, use "mps"
    elif platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"

    # For non-Windows and non-M1 Mac systems, use "cuda" if available, otherwise use "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

# Added function to check for Coral USB Accelerator
def is_coral_available():
    return platform.machine() == "aarch64" and "USB Accelerator" in platform.uname().release

def substitute_env_variables(input_path):
    # Generate the output filename
    output_path = os.path.splitext(input_path)[0] + '_substituted.yaml'

    # Read the input .yaml file
    with open(input_path, 'r') as input_file:
        content = input_file.read()

    # Substitute environment variables using string.Template
    template = string.Template(content)
    substituted_content = template.substitute(os.environ)

    # Write the substituted content to the output .yaml file
    with open(output_path, 'w') as output_file:
        output_file.write(substituted_content)

    return output_path