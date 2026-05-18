import sys

import torch


_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from sf_examples.vizdoom.enjoy_vizdoom import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
