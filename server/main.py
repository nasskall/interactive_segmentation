"""Start the REST backend with the default RITM checkpoint.

    python -m server.main [--host 127.0.0.1] [--port 8000] [--cpu]
                          [--ssl-certfile cert.pem --ssl-keyfile key.pem]

Serving beyond this machine (--host other than 127.0.0.1 / localhost)
should use HTTPS: pass a certificate and key, e.g. from the lab's internal
CA. Without them the server still starts but warns, because passwords and
tokens would cross the network in clear text.
"""

import argparse
import sys
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
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data',
                        help='Where labelling projects and their images are stored.')
    parser.add_argument('--ssl-certfile', type=Path, help='TLS certificate (PEM) for HTTPS.')
    parser.add_argument('--ssl-keyfile', type=Path, help='TLS private key (PEM) for HTTPS.')
    args = parser.parse_args()
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        parser.error('--ssl-certfile and --ssl-keyfile go together')
    tls = args.ssl_certfile is not None
    if not tls and args.host not in ('127.0.0.1', 'localhost', '::1'):
        print(f'WARNING: serving on {args.host} over plain HTTP. Passwords and tokens will '
              'cross the network unencrypted; pass --ssl-certfile/--ssl-keyfile.',
              file=sys.stderr, flush=True)

    torch.backends.cudnn.deterministic = True
    device = 'cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = utils.load_is_model(str(WEIGHTS_DIR / DEFAULT_CHECKPOINT), device=device,
                                cpu_dist_maps=True)
    model._model_type = 'ritm'

    manager = ModelManager(model, device, 'RITM', DEFAULT_CHECKPOINT, WEIGHTS_DIR)
    app = create_app(manager, static_dir=ROOT / 'frontend' / 'dist', data_dir=args.data_dir)
    if app.state.labelling.setup_token:
        print('\nNo users yet. Create the first admin with this one-time setup token:\n'
              f'    {app.state.labelling.setup_token}\n'
              'e.g. POST /api/auth/setup (see /docs). It changes on every restart.\n',
              flush=True)
    uvicorn.run(app, host=args.host, port=args.port,
                ssl_certfile=str(args.ssl_certfile) if tls else None,
                ssl_keyfile=str(args.ssl_keyfile) if tls else None)


if __name__ == '__main__':
    main()
