import os
import tqdm
import torch
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch_geometric.data import DataLoader

from dagr.utils.args import FLOPS_FLAGS
from dagr.utils.buffers import DictBuffer, format_data

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR

from dagr.asynchronous.evaluate_flops import evaluate_flops


if __name__ == '__main__':
    import torch_geometric
    seed = 42
    torch_geometric.seed.seed_everything(seed)
    args = FLOPS_FLAGS()
    assert "checkpoint" in args

    project = f"flops-{args.dataset}-{args.task}"
    pbar = tqdm.tqdm(total=4)

    pbar.set_description("Loading dataset")
    dataset_path = args.dataset_directory / args.dataset
    print("init datasets")
    dataset = DSEC(args.dataset_directory, "test", Augmentations.transform_testing, debug=True, min_bbox_diag=15, min_bbox_height=10)
    loader = DataLoader(dataset, follow_batch=['bbox', "bbox0"], batch_size=args.batch_size, shuffle=False, num_workers=16)
    pbar.update(1)

    pbar.set_description("Initializing net")
    model = DAGR(args, height=dataset.height, width=dataset.width)
    model = model.cuda()
    model.eval()
    pbar.update(1)

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['ema'])
    pbar.update(1)

    model.cache_luts(radius=args.radius, height=dataset.height, width=dataset.width)

    pbar.set_description("Computing FLOPS")
    buffer = DictBuffer()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    pbar_flops = tqdm.tqdm(total=len(loader.dataset), desc="Computing FLOPS")
    for i, data in enumerate(loader):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        flops_evaluation = evaluate_flops(model, data,
                                          check_consistency=args.check_consistency,
                                          return_all_samples=True, dense=args.dense)
        if flops_evaluation is None:
            continue

        buffer.update(flops_evaluation['flops_per_layer'])
        buffer.save(args.output_directory / "flops_per_layer.pth")
        tot_flops = sum(buffer.compute().values())

        pbar_flops.set_description(f"Total FLOPS {tot_flops}")
        pbar_flops.update(1)

    print(sum(buffer.compute().values()))
    pbar.update(1)




