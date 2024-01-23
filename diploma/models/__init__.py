from . import octoai_llms

MODEL_REGISTRY = {
    "octoai": octoai_llms.OctoAIEndpointLM,
}

def get_model(model_name):
    return MODEL_REGISTRY[model_name]
