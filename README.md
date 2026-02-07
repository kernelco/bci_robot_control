# BCI Robot Control Template

This template users the Kernel SDK to stream TD-NIRS and EEG data into a real-time data pipeline.

The data pipeline uses the NVIDIA Holoscan SDK and has 4 operators:

```
[NirsStreamOperator]--
                      |->[WindowOperator]->[InferenceOperator]
[EegStreamOperator]---
```

## NirsStreamOperator

Subscribes to TD-NIRS moments data using the Kernel SDK.

## EegStreamOperator

Subscribes to EEG voltage data using the Kernel SDK.

## WindowOperator

Aggregates the TD-NIRS moments by module/source-detector-separation bucket.

Buffers until it has 15s of data and emits a struct containing 15s of aggregated TD-NIRS moments and EEG voltage.

## InferenceOperator

This is where you can infer on the data and take action.