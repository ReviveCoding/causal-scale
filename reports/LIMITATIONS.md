# Limitations

- Criteo is an offline randomized advertising benchmark; policy value is neither observed online revenue nor production ROI.
- The authentic 13.98M run is a systems workload and is not an independent causal holdout.
- Local Spark workers share one physical Windows machine and are not distributed physical nodes.
- CUDA speed and CPU/GPU prediction agreement are systems properties, not evidence that GPU fitting is causally superior.
- Twins is a benchmark construction around paired twins and does not establish universal external validity.
- DragonNet, full IHDP replications, Hillstrom, ONNX CUDA, and physical multi-node scaling are optional and may be resource-skipped.
