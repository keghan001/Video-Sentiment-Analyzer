from sagemaker.pytorch import PyTorchModel #type:ignore
import sagemaker
from dotenv import load_dotenv
import os

load_dotenv()

def deploy_endpoint():
    sagemaker.Session() #type: ignore
    role = os.getenv("ENDPOINT_ROLE")
    
    try: 
        model_uri = os.getenv("MODEL_URI", "None")
    except ValueError as e:
        print("Model uri is not available")
    
    model = PyTorchModel(
        model_data=model_uri,
        entry_point="inference.py",
        source_dir=".",
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        name="sentiment-analyzer-model"
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analyzer-endpoint"
    )
    

if __name__ == "__main__":
    deploy_endpoint()