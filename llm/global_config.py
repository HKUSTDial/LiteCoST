# config.py

# Default model value
_model = 'gpt-4o'

def set_model(new_model):
    """
    Set the global model value.

    Args:
        new_model (str): The model to set globally. Should be 'gpt', 'llama', or 'deepseek'.
    """
    global _model
    _model = new_model

def get_model():
    """
    Get the current global model value.

    Returns:
        str: The current global model value.
    """
    return _model
