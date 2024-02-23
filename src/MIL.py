
import os
import sys
from omegaconf import OmegaConf
import argparse
import pkgutil
import importlib

import steps

def main():
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    
    STEPS = list(map(lambda i: i.name, pkgutil.iter_modules(steps.__path__))) # pyright: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c', required=True, type=str)
    parser.add_argument('--show-config', action='store_true', default=False)
    parser.add_argument('--overrides','-o', nargs='+', default=[])
    parser.add_argument('--step', '-s', choices=STEPS, default=None)
    args = parser.parse_args()

    try:
        c1 = OmegaConf.load(args.config)
        c2 = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(c1, c2)
        #OmegaConf.resolve(cfg)
    except Exception as e:
        print(f'could not read config: {args.config} {args.overrides}')
        print(e)
        sys.exit(1)

    if args.show_config:
        print(OmegaConf.to_yaml(cfg))
        sys.exit(0)
    elif args.step is None:
        parser.print_usage()
        sys.exit(1)

    m = importlib.import_module(f'steps.{args.step}')
    m.main(cfg)

if __name__ == '__main__':
    main()

