# ECCV2022/dis-background-removal Cog Model

Inspired by the huggingface space[ECCV2022/dis-background-removal](https://huggingface.co/spaces/ECCV2022/dis-background-removal), this is an implementation of [ECCV2022/dis-background-removal](https://github.com/xuebinqin/DIS) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

# Setup

Checkpoints file `isnet.pth` and `data_loader_cache` script are from the original repository: [xuebinqin/DIS](https://github.com/xuebinqin/DIS)

## Basic Usage

To run a prediction:

    cog predict -i image=@cologne.png

| Input | Output | Mask |
|-------|--------|------|
| ![Input](cologne.png) | ![Output](output.0.png) | ![Mask](output.1.png) |

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

`models/` and `data_loader_cache.py` come from [xuebinqin/DIS](https://github.com/xuebinqin/DIS), which is Apache 2.0.

The model weights are not covered by this license. [xuebinqin/DIS](https://github.com/xuebinqin/DIS) is released under the [Apache 2.0](https://github.com/xuebinqin/DIS/blob/main/LICENSE.md) license.
