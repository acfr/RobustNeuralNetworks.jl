## Testing `RobustNeuralNetworks.jl` on a GPU

There is currently full support for using models from `RobustNeuralNetworks.jl` on a GPU. However, the speed could definitely be improved with some code optimisations, and we don't have any CI testing on the GPU.

The scripts in this directory serve two purposes:
- They provide a means of benchmarking model performance on a GPU
- They act as unit tests to verify the models can be trained on a GPU

There is an [open issue](https://github.com/acfr/RobustNeuralNetworks.jl/issues/119) on improving the speed of our models on GPUs. Any and all contributions are welcome.