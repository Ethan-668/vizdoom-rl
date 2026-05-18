import sys

import torch


_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

import sf_examples.vizdoom.doom.doom_utils as doom_utils  # noqa: E402


_original_make_doom_env_impl = doom_utils.make_doom_env_impl


def _make_doom_env_impl_hd(*args, **kwargs):
    if kwargs.get("custom_resolution") is None:
        kwargs["custom_resolution"] = "640x480"
    return _original_make_doom_env_impl(*args, **kwargs)


doom_utils.make_doom_env_impl = _make_doom_env_impl_hd

from sf_examples.vizdoom import train_vizdoom  # noqa: E402


train_vizdoom.make_doom_env_from_spec.__globals__["make_doom_env_impl"] = _make_doom_env_impl_hd

from sf_examples.vizdoom.enjoy_vizdoom import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
