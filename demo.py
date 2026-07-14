import tkinter as tk
import os, sys
import torch

from interactive_demo.app import InteractiveDemoApp
from isegm.inference import utils


def get_config_path():
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, use the _MEIPASS directory
        base_path = sys._MEIPASS
    else:
        # If running as a normal script, use the current directory
        base_path = os.path.abspath(".")

    return os.path.join(base_path, 'config.yml')


def get_model_path():
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, use the _MEIPASS directory
        base_path = sys._MEIPASS
    else:
        # If running as a normal script, use the current directory
        base_path = os.path.abspath(".") + '/model_weights'

    return os.path.join(base_path, 'best_checkpoint_068.pth')

def main():
    #args, cfg = parse_args()
    #print(f"Running from: {os.getcwd()}")
    #print(f"Resolved model path: {get_model_path()}")
    #cfg_path = get_config_path()
    #cfg = exp.load_config_file(cfg_path, return_edict=True)
    torch.backends.cudnn.deterministic = True
    # #checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    mdl_path = get_model_path()
    model = utils.load_is_model(mdl_path, device=device, cpu_dist_maps=True)
    model._model_type = 'ritm'
    limit_longest_size = 5000
    root = tk.Tk()
    root.minsize(1060, 780)
    app = InteractiveDemoApp(root, limit_longest_size, device, model)
    root.deiconify()
    app.mainloop()


# def parse_args():
#     parser = argparse.ArgumentParser()

#     # parser.add_argument('--checkpoint', type=str, required=True,
#     #                     help='The path to the checkpoint. '
#     #                          'This can be a relative path (rerun lative to cfg.INTERACTIVE_MODELS_PATH) '
#     #                          'or an absolute path. The file extension can be omitted.')

#     parser.add_argument('--gpu', type=int, default=0,
#                         help='Id of GPU to use.')

#     parser.add_argument('--cpu', action='store_true', default=False,
#                         help='Use only CPU for inference.')

#     parser.add_argument('--limit-longest-size', type=int, default=800,
#                         help='If the largest side of an image exceeds this value, '
#                              'it is resized so that its largest side is equal to this value.')

#     parser.add_argument('--cfg', type=str, default="config.yml",
#                         help='The path to the config file.')

#     args = parser.parse_args()
#     # if args.cpu:
#     #     args.device = torch.device('cpu')
#     # else:
#     #     args.device = torch.device(f'cuda:{args.gpu}')
#     cfg = exp.load_config_file(args.cfg, return_edict=True)

#     return args, cfg


if __name__ == '__main__':
    main()
