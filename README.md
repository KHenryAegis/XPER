# XPER

Code and data for *Explore Once, Reuse Across Models? When Stored Agent Experience Pays Off for a New Model*.

When can a new model benefit from another agent’s tool interactions? XPER compares reusing stored trajectories with collecting experience directly on the target model. The experiments cover telecom support tickets in τ²-bench and integer-answer math problems solved with a Python tool.

## Run the experiments

The paper used one Linux node with eight NVIDIA H20 GPUs (96 GB each). Follow the [GPU setup instructions](experiments/README.md#prepare-the-environment-and-inputs) to install dependencies and prepare the model checkpoints and datasets. Then run the following command from the repository root.

```bash
bash experiments/run_gpu.sh
```

To preview the execution order, use `bash experiments/run_gpu.sh --list`. To reproduce a specific result, find its script and prerequisites in the [experiment index](experiments/README.md#run-on-gpus). A full run writes the rerun evaluation summaries to `outputs/gpu/eval_summaries.csv`.

## Recompute the reported results on CPU

```bash
python -m pip install numpy==2.2.6
python scripts/reproduce.py
```

The script reads the released episode and training-round records and recomputes success rates, task-family breakdowns, validation-probe decisions and paired intervals in `outputs/`. The recorded weights and GPU prerequisites are listed in [the reproduction guide](experiments/README.md#prepare-the-environment-and-inputs).

## Repository layout

| Directory | Contents |
|---|---|
| [experiments/](experiments/) | GPU launcher, individual experiment scripts, and reproduction guide |
| [xper/](xper/) | Agent environments, tool actions, and trajectory storage |
| [scripts/](scripts/) | Training, evaluation, and analysis drivers |
| [data/](data/) | Experiment records, reference results, and input manifests |
| [tools/](tools/) | Benchmark setup, input checks, and result export |
| [tests/](tests/) | Tests for the experiment code |

## Tests

With the experiment environment active, run:

```bash
python -m pip install pytest jsonschema
python -m pytest -rs tests
```

## License

The project code is distributed under the [MIT License](LICENSE). Third-party datasets, benchmarks, and model weights retain their respective licenses.
