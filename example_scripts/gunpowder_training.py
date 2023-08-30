from funlib.learn.torch.models import UNet, ConvPass
import logging
import math
import torch
import multiprocessing
multiprocessing.set_start_method("fork")
import gunpowder as gp

logging.basicConfig(level=logging.INFO)

num_training_images = 100
batch_size = 32
patch_shape = (252, 252)
learning_rate = 1e-4


def train_until(max_iterations):

    # create model, loss, and optimizer

    unet = UNet(
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=4,
        downsample_factors=[(2, 2), (2, 2), (2, 2)],
        kernel_size_down=[[[3, 3], [3, 3]]]*4,
        kernel_size_up=[[[3, 3], [3, 3]]]*3,
        padding='valid',
        constant_upsample=True)
    model = torch.nn.Sequential(
        unet,
        ConvPass(12, 1, [(1, 1)], activation=None),
        torch.nn.Sigmoid())
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # assemble gunpowder training pipeline

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    prediction = gp.ArrayKey('PREDICTION')

    sources = tuple(
        gp.ZarrSource(
            'neurons.zarr',  # the zarr container
            {raw: f'raw_{i}', labels: f'labels_{i}'},  # which dataset to associate to the array key
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False)
            }
        )
        for i in range(num_training_images)
    )
    source = sources + gp.RandomProvider()
    random_location = gp.RandomLocation(
        min_masked=0.1,
        mask=labels)

    normalize_raw = gp.Normalize(raw)
    normalize_labels = gp.Normalize(labels, factor=1.0/255.0)  # ensure labels are float

    simple_augment = gp.SimpleAugment()
    elastic_augment = gp.ElasticAugment(
        control_point_spacing=(16, 16),
        jitter_sigma=(1.0, 1.0),
        rotation_interval=(0, math.pi/2))
    intensity_augment = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.01,
        shift_max=0.01)
    unsqueeze = gp.Unsqueeze([raw, labels], axis=0)  # add "channel" dim
    stack = gp.Stack(batch_size)
    precache = gp.PreCache(num_workers=4)

    train = gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs = {
          'input': raw
        },
        outputs = {
          0: prediction
        },
        loss_inputs = {
          0: prediction,
          1: labels
        },
        save_every=1000)

    snapshot = gp.Snapshot(
        {
            raw: 'raw',
            prediction: 'prediction',
            labels: 'labels'
        },
        output_filename='iteration_{iteration}.zarr',
        every=100)

    pipeline = (
        source +
        random_location +
        normalize_raw +
        normalize_labels +
        simple_augment +
        elastic_augment +
        intensity_augment +
        unsqueeze +
        stack +
        precache +
        train +
        snapshot)

    output_shape = tuple(model(torch.zeros((1, 1) + patch_shape)).shape[2:])
    print("Input shape:", patch_shape)
    print("Output shape:", output_shape)

    request = gp.BatchRequest()
    request.add(raw, patch_shape)
    request.add(labels, output_shape)
    request.add(prediction, output_shape)

    with gp.build(pipeline):
        for i in range(max_iterations):
            batch = pipeline.request_batch(request)

if __name__ == '__main__':

    train_until(10000)
