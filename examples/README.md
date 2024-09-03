# gentun examples

This directory contains examples of how to use the `gentun` package to optimize model hyperparameters.
Be sure to [install the gentun package](../README.md#installation) and download the required datasets
using [the script provided](./get_datasets.sh) before running these examples.

## Sample distributed algorithm

To run this example, [setup your Redis server](../README.md#redis-setup) in addition to the previous
steps. Next, start the controller node:

```python
python sample_controller.py
```

And, for evary worker node, start your worker code:

```python
python sample_worker.py
```
