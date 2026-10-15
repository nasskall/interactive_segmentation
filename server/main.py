"""Start the REST backend with the default RITM checkpoint.

    python -m server.main [--host 127.0.0.1] [--port 8000] [--cpu]
"""

import argparse
from pathlib import Path

import torch
import uvicorn

from isegm.inference import utils
from server.app import create_app
from server.sessions import ModelManager

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / 'model_weights'
DEFAULT_CHECKPOINT = 'best_checkpoint_068.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--cpu', action='store_true', help='Run inference on CPU.')
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    device = 'cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = utils.load_is_model(str(WEIGHTS_DIR / DEFAULT_CHECKPOINT), device=device,
                                cpu_dist_maps=True)
    model._model_type = 'ritm'

    manager = ModelManager(model, device, 'RITM', DEFAULT_CHECKPOINT, WEIGHTS_DIR)
    app = create_app(manager, static_dir=ROOT / 'frontend' / 'dist')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
