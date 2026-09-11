import torch
import time
from thop import profile
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict
import sys

sys.path.append('.')
from models.DAPointMamba import DAMamba

def get_model(config_path, device):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = EasyDict(config_dict)
    model = DAMamba(config.model).to(device)
    model.eval()
    return model

def profile_model(model, num_points, batch_size=1, device='cuda'):
    pts = torch.randn(batch_size, num_points, 3).to(device)
    # FLOPs & Params
    flops, params = profile(model, inputs=(pts,), verbose=False)
    # Inference Time
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(pts)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = model(pts)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 50
    return flops, params, elapsed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model = get_model(args.config, device)

    point_nums = [512, 1024, 2048, 4096]
    flops_list, params_list, time_list = [], [], []

    for n in point_nums:
        print(f'Profiling for {n} points...')
        flops, params, inf_time = profile_model(model, n, batch_size=1, device=device)
        flops_list.append(flops / 1e9)  # GFLOPs
        params_list.append(params / 1e6)  # M
        time_list.append(inf_time * 1000)  # ms
        print(f'Points: {n}, FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M, Time: {inf_time*1000:.2f}ms')

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(point_nums, flops_list, marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('FLOPs (G)')
    plt.title('FLOPs vs Points')

    plt.subplot(1,3,2)
    plt.plot(point_nums, params_list, marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('Parameters (M)')
    plt.title('Parameters vs Points')

    plt.subplot(1,3,3)
    plt.plot(point_nums, time_list, marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time vs Points')

    plt.tight_layout()
    plt.savefig('profile_results.png')
    plt.show()

if __name__ == '__main__':
    main() 