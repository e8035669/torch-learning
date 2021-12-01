import os
import base64
import yaml
import torch

import mlflow
import mlflow.keras
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env

def save_pytorch_model(pytorch_model, artifact_path, image_dims, domain):
    with TempDir() as tmp:
        data_path = tmp.path('image_model')
        os.mkdir(data_path)
        conf = {
            'image_dims': '/'.join(map(str, image_dims)),
            'domain': '/'.join(map(str, domain)),
        }
        with open(os.path.join(data_path, 'conf.yaml'), 'w') as f:
            yaml.safe_dump(conf, stream=f)
        torch_path = os.path.join(data_path, 'torch_model')
        mlflow.pytorch.save_model(pytorch_model, path=torch_path)
        conda_env = tmp.path('conda_env.yaml')
        with open(conda_env, 'w') as f:
            yaml.safe_dump(mlflow.pytorch.get_default_conda_env(), stream=f)
            
        mlflow.pyfunc.save_model(
            artifact_path,
            loader_module=__name__,
            code_path=[__file__],
            data_path=data_path,
            conda_env=conda_env
        )

def _load_pyfunc(path):
    print('path', path)
    with open(os.path.join(path, 'conf.yaml'), 'r') as f:
        conf = yaml.safe_load(f)
    print('conf', conf)
    
    model_path = os.path.join(path, 'torch_model')
    print('model_path', model_path)
    
    net = mlflow.pytorch.load_model(model_path)
    print(net)
    
    return net

