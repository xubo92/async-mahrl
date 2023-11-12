# async-mahrl
Official code for the paper "Asynchronous, Option-Based Multi-Agent Policy Gradient: A Conditional Reasoning Approach"

## Supported Environments
- Water-Filling 
- Tool-Delivery
- Capture-Target

## Usage
### Preparation
1. Install conda and create a conda environment (python 3.6.8) \
     `conda create env --name mahrl `
2. Install necessary dependencies \
     `pip install -r requirements.txt`
3. Validate the installation of ai2thor simulator
    * On Windows, you may want to check: https://github.com/allenai/ai2thor/issues/811.
    * On Linux, you may want to check: https://github.com/allenai/ai2thor-docker. You don't necessarily use the docker, but startx() and an minimal example is helpful to run the simulator on a Linux server without a monitor.
    * Anyway, run this example to see if everything about ai2thor works well. 
    ``` import ai2thor.controller
    import ai2thor.platform
    from pprint import pprint

    if __name__ == '__main__':
        controller = ai2thor.controller.Controller(platform=ai2thor.platform.CloudRendering, scene='FloorPlan28')
        event = controller.step(action='RotateRight')
        pprint(event.metadata['agent'])```
### Training
* Water-Filling task \
`python train_wf.py --scheme [fully-dec, partial-dec, fully-cen, partial-cen, sync-cut, sync-wait, end2end] --seed [seed]`
* Tool-Delivery task \
`python tool_delivery/train_td.py --scheme [fully-dec, partial-dec, fully-cen, partial-cen, sync-cut,sync-wait] --seed [seed]`
* Capture-Target task \
`python capture_target/train_ct.py --scheme [fully-dec,  fully-cen] --seed [seed]`

### Evaluation
Use the same commands above, but add one more line in `train_wf.py`, `train_td.py` or `train_ct.py` as follows. \
For example,  in `train_td.py` line 95, add `all_args.model_dir = "./results/toolDeliverySeparate/fully-dec/mappo/mlp/run10/models"`, and change the model saving path to yours.

### Cite the paper
```
@article{lyu2022asynchronous,
title={Asynchronous, Option-Based Multi-Agent Policy Gradient: A Conditional Reasoning Approach},
author={Lyu, Xubo and Banitalebi-Dehkordi, Amin and Chen, Mo and Zhang, Yong},
journal={arXiv preprint arXiv:2203.15925},
year={2022}
}
```


