from io import BytesIO
import os
import base64
import yaml
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from PIL import Image
import pandas as pd
import mlflow
import mlflow.keras
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env

def get_pytorch_env_patch():
    e = mlflow.pytorch.get_default_conda_env()
    e['channels'].append('pytorch')
    e['dependencies'].extend(['pytorch', 'torchvision', 'torchaudio', 'cudatoolkit=11.3'])
    find_pip = tuple(filter(lambda p: isinstance(p, dict) and 'pip' in p, e['dependencies']))
    find_torch = tuple(filter(lambda p: 'torch' in p, find_pip[0]['pip']))
    for p in find_torch:
        find_pip[0]['pip'].remove(p)
    find_python = tuple(filter(lambda p: 'python' in p, e['dependencies']))
    e['dependencies'].remove(find_python[0])
    e['dependencies'].append('python')
    return e

def save_pytorch_model(pytorch_model, artifact_path, image_dims, domain, batch_limit = 16):
    with TempDir() as tmp:
        data_path = tmp.path('image_model')
        os.mkdir(data_path)
        conf = {
            'image_dims': '/'.join(map(str, image_dims)),
            'domain': '/'.join(map(str, domain)),
            'batch_limit': batch_limit,
        }
        with open(os.path.join(data_path, 'conf.yaml'), 'w') as f:
            yaml.safe_dump(conf, stream=f)
        torch_path = os.path.join(data_path, 'torch_model')
        mlflow.pytorch.save_model(pytorch_model, path=torch_path, conda_env=get_pytorch_env_patch())
        conda_env = tmp.path('conda_env.yaml')
        with open(conda_env, 'w') as f:
            yaml.safe_dump(get_pytorch_env_patch(), stream=f)

        mlflow.pyfunc.save_model(
            artifact_path,
            loader_module=__name__,
            code_path=[__file__],
            data_path=data_path,
            conda_env=conda_env
        )

def load_image(img_data: bytes):
    with BytesIO(img_data) as fd:
        img = Image.open(fd)
        img.load()
    return img


class PytorchClassifierWrapper:
    def __init__(self, model, device, labels, dims, batch_limit):
        self.model = model
        self.device = device
        self.labels = labels
        self.dims = dims
        self.batch_limit = batch_limit
        if dims[1] == 3:
            self.transforms = T.Compose([
                T.Resize(dims[2:]),
                T.ToTensor()
            ])
        else:
            self.transforms = T.Compose([
                T.Grayscale(),
                T.Resize(dims[2:]),
                T.ToTensor()
            ])


    def predict(self, data: pd.DataFrame):
        data = data.values.reshape(-1)
        data = [load_image(base64.decodebytes(bytearray(img, encoding='utf-8'))) for img in data]
        data = torch.cat([torch.unsqueeze(self.transforms(img), 0) for img in data])
        chunks = torch.split(data, self.batch_limit, 0)
        print(data.shape)

        confidences = []
        classes = []

        with torch.no_grad():
            for x in chunks:
                x = x.to(self.device)
                y = self.model(x)
                y = F.softmax(y, 1)
                conf, cls = y.max(1)
                confidences.append(conf)
                classes.append(cls)

        confidences = torch.cat(confidences, 0)
        classes = torch.cat(classes, 0)

        confidences = confidences.cpu().numpy()
        classes = classes.cpu().numpy()
        cls_name = [ self.labels[i] if i < len(self.labels) else "" for i in classes]

        result = [{
            "conf": c,
            "cls": l,
            "cls_name": n
        } for c, l, n in zip(confidences, classes, cls_name)]

        print(result)

        return result


def _load_pyfunc(path):
    print('path', path)
    with open(os.path.join(path, 'conf.yaml'), 'r') as f:
        conf = yaml.safe_load(f)
    print('conf', conf)

    labels = conf['domain'].split('/')
    dims = [int(i) for i in conf['image_dims'].split('/')]
    batch_limit = int(conf['batch_limit'])

    model_path = os.path.join(path, 'torch_model')
    print('model_path', model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    net = mlflow.pytorch.load_model(model_path, map_location=device)
    net.eval()
    print(net)

    return PytorchClassifierWrapper(net, device, labels, dims, batch_limit)

