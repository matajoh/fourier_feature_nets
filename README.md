# Fourier Feature Networks and their Applications to Neural Volume Rendering

This repository is a companion to [TODO: lecture name], which is available for
viewing at [TODO: video URL]. In it you will find Jupyter notebooks for:

1. 1D Signal Reconstruction [TODO: link]
2. 2D Image Regression [TODO: link]
3. Voxel-based Volume Rendering [TODO: link]
4. 3D Volume Rendering with NeRF [TODO: link]

You will also find scripts that will allow you to run larger scale experiments
outside of a Jupyter Notebook.

# Getting Started

In this section I will outline how to run the various experiments. Before I
begin, it is worth noting that while the defaults are all reasonable and will
produce the results you see in the lecture, it can be very educational to
play around with different hyperparameter values and observe the results.

In order to run the various experiments, you will first need to install
the requirements for the repository, ideally in a virtual environment. As this
code heavily relies upon PyTorch, you should install the correct version for
your platform. The guide [here](https://pytorch.org/get-started/locally/)
is very useful and I suggest you follow it closely. Once that is done, you
can run the following:

```
pip install wheel
pip install -r requirements.txt
```

You should now be ready to run any of the experiment scripts in this
repository.

# Fourier Feature Networks

TODO description and diagram

# Experiments

## 2D Image Regression

To get started with 2D Image Regression, run the following command:

    python train_image_regression.py data/cat.jpg mlp outputs/cat_mlp

A window should pop up as the system trains that looks like this:

![Image Regression](docs/image_regression.jpg)

At the end it will show you the result, which as you will have come to
expect from the lecture is severaly lacking in detail due to the lack
of high-frequency gradients. Try running the same script with
`positional` or `gaussian` in place of `mlp` to see how using
Fourier features dramatically improves the quality. Your results should
look like what you see below:

Feel free to pass the script your own images and see what happens!

## Ray Sampling

As a preparation for working with volume rendering, it can be useful to get a
feel for the training data. If you run:

    python test_ray_sampling.py lego_400.npz lego_400_rays.html

This should download the dataset into the `data` directory and then create
a scenepic showing what the ray sampling data looks like. Notice how the rays
pass from the camera through the pixels and into the volume. Try running
this script again with `--stratified` to see what happens when we add some
uniform noise to the samples.

> NOTE: Any of the following scripts can also be run with the other provided
> dataset, `antinous_400.npz`. If you want to provide your own dataset,
> see the instructions in [Bring Your Own Data](#bring-your-own-data).

## Voxel-based Volume Rendering

Just like in the lecture, we'll start with voxel-based rendering. If you run
the following command:

    python train_voxels.py lego_400.npz 128 outputs/lego_400_vox128

You may have trouble running this script (and the ones that follow) if your
computer does not have a GPU with enough memory. See
[Running on Azure ML](#running-on-azure-ml) for information on how to run these
experiments in the cloud.

If you look in the `train` and `val` folders in the output
directory you can see images produced during training showing how
the model improves over time. There is also a visualization of the
model provided in the `voxels.html` scenepic.

Another way to visualize what the model has learned is to produce a
voxelization of the model. This is different from the voxel-based volume
rendering, in which multiple voxels contribute to a single sample. Rather, it
is a sparse octree containing voxels at the places the model has determined are
solid, thus providing a rough sense of how the model is producing the rendered
images. You can produce a scenepic showing this via the following command:

    python voxelize_model.py outputs/lego_400_vox128/voxels.pt lego_400.npz lego_400_voxels.html

This will work for any of the volumetric rendering models.

## Tiny NeRF

The first neural rendering technique we looked at was so-called "Tiny" NeRF, in
which the view direction is not incorporated but we only focus on the 3D
position within the volume. You can train Tiny NeRF models using the following
command:

    python train_tiny_nerf.py lego_400.npz mlp outputs/lego_400_mlp/

Substituting `positional` and `gaussian` as before to try out different modes
of Fourier encoding. You'll notice again the same low-resolution results for
MLP and similarly improved results when Fourier features are introduced.

# NeRF

In the results above you possibly noticed that specularities and transparency
were not quite right. This is because those effects require the incorporation
of the *view direction*, that is, where the camera is located in relation to
the position. NeRF introduces this via a novel structure in the fairly simple
model we've used so far:

TODO NeRF model diagram

The other major difference from what has come before is that NeRF samples
the volume in a different way. The original paper
(see [here](https://www.matthewtancik.com/nerf)) performs two-tiers of sampling.
First, they sample a *coarse* network, which determines where in the space
is opaque, and then they use that to create a second set of samples which
are used to train a *fine* network. For the purpose of this lecture, we do
something very similar in spirit, which is to use the voxel model we trained
above as the `coarse` model. You can see how this changes the sampling of the
volume by running the `test_ray_sampling.py` script again:

    python test_ray_sampling.py lego_400.npz lego_400_fine.html --opacity-model lego_400_vox128.pt

You should now be able to see how additional samples are clustering near
the location of the model, as opposed to being only evenly distributed over
the volume. This helps the NeRF model to learn detail. Try passing in
`--stratified` again to see the effects for random sampling as well.

> NOTE: the Tiny NeRF model can also take advantage of fine sampling using an
> opacity model. Try it out!

You can train the NeRF model with the following command:

   python train_nerf.py lego_400.npz outputs/lego_400_nerf --opacity-model lego_400_vox128.pt

While this model can train for many more steps than 50000 and continue to
improve, you should already be able to see the increase in quality over the
other models from adding in view direction.

# Running on Azure ML

It is outside of the scope of this lecture (or repository) to describe in detail
how to get access to cloud computing resources for machine learning via
Azure ML. However, there are some amazing resources out there already.
For the purpose of this repository, all you need to do is complete
[this Quickstart Tutorial](https://docs.microsoft.com/en-gb/azure/machine-learning/quickstart-create-resources)
and download the `config.json` associated with your workspace into the root
of the repository. You can then run any of the training scripts in Azure ML
using the `submit_aml_run.py` script, like so:

    python submit_aml_run.py cat <compute> train_image_regression.py "cat.jpg mlp outputs"

Where `cat` is the experiment name (you can choose anything here) that will
group different runs together, and where you replace `<compute>` with the
name of the compute target you want to use to run the experiment (which
will need to have a GPU available). Finally you provide the script name
(in this case, `train_image_regression.py`, which I suggest you use while you
are getting your workspace up and running) and the arguments to the script as
a string. If you get an error, make certain you've run:

    pip install -r requirements-azureml.txt

If everything is working, you should receive a link that lets you monitor
the experiment and view the output images and results in your browser.

# Bring Your Own Data

For image regression, the script should run on any image you have that is
greater than or equal to the resolution you provide (default is 512x512).
It will center-crop the image first, and then scale it as needed.

For 3D datasets, the requirement is somewhat higher. The easiest way
is to use the tool I've provided [here](TODO) which, among other things,
can take a 3D model and use it to produce a volume rendering dataset
in the expected file format. However, if you wish to build your own
(or adapt an existing dataset) the format is fairly straightforward. Build
an NPZ file which has the following tensors:

| Name         |     Shape    |  dtype  | description |
|--------------|:------------:|:-------:|-------------|
| images       | (C, D, D, 4) |  uint8  | Tensor of camera images with RGBA pixel values.
| intrinsics   |   (C, 3, 3)  | float32 | Tensor of camera intrinsics (i.e. projection) matrices
| extrinsics   |   (C, 4, 4)  | float32 | Tensor of camera extrinsics (i.e. camera to world) matrices
| bounds       |    (4, 4)    | float32 | Rigid transform indicating the rough bounds of the volume to be rendered. Does not need to be exact, as it is used purely for visualizations.
| split_counts |      (3)     |  int32  | Number of cameras (in order) for train, val and test data splits.

where `C` is the number of cameras and `D` is the image resolution.
