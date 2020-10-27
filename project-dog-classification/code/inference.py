import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import pickle

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'model_transfer.pt')
    with torch.neo.config(model_dir=model_dir, neo_runtime=True):
        model = torch.jit.load(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # We recommend that you run warm-up inference during model load
        sample_input_path = os.path.join(model_dir, 'sample_input.pkl')
        with open(sample_input_path, 'rb') as input_file:
            model_input = pickle.load(input_file)
        if torch.is_tensor(model_input):
            model_input = model_input.to(device)
            model(model_input)
        elif isinstance(model_input, tuple):
            model_input = (inp.to(device)
                           for inp in model_input if torch.is_tensor(inp))
            model(*model_input)
        else:
            print("Only supports a torch tensor or a tuple of torch tensors")

        return model

def transform_fn(model, request_body, request_content_type,
                 response_content_type):
    
    decoded = Image.open(io.BytesIO(request_body))
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    normalized = preprocess(decoded)
    batchified = normalized.unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchified = batchified.to(device)
    
    output = model.forward(batchified)

    return json.dumps(output.cpu().numpy().tolist()), response_content_type
