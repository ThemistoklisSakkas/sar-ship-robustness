# Deployment notes: ONNX now, TensorRT on NVIDIA Jetson later

## What this repository delivers

- `best.pt` (PyTorch) and `best.onnx` (FP32, static 1×3×640×640 input, opset chosen by
  Ultralytics, graph simplified) for the selected model.
- An equivalence check: the ONNX model must reproduce the PyTorch test mAP@0.5:0.95 within
  0.002 (`python -m sarship.benchmark`).
- Batch-1 latency on the Colab T4 GPU and CPU. These numbers are **not** Jetson numbers.

## Moving to a Jetson (Orin Nano / NX / AGX) with TensorRT

1. **Build the engine on the target device.** A TensorRT engine is specific to the GPU
   architecture and to the TensorRT/JetPack version, so it is built from the ONNX file on the
   Jetson itself (`yolo export format=engine` or `trtexec --onnx=best.onnx`), never copied from
   another machine. Pin the JetPack version in the deployment record.
2. **Precision.**
   - FP16: usually close to FP32 accuracy and a large speed-up; re-check it anyway.
   - INT8: needs a calibration set of representative SAR chips (use the validation images, not
     the test set) and can cost accuracy on small, low-contrast targets — exactly our hard cases.
3. **Re-validate after conversion.** Run the same test-set evaluation and the paired comparison
   (FP32 PyTorch vs the TensorRT engine) and accept the engine only if the loss is within an
   agreed tolerance, as done here for ONNX. Include the degraded test conditions: quantisation
   errors and degradations can compound.
4. **Measure on the device, in the deployment configuration:** power mode (`nvpmodel`), clocks
   (`jetson_clocks`), thermal throttling over a sustained run, batch size, and the full
   pipeline (image decode + pre-processing + inference + NMS), not only the network.
5. **Input quality gate.** Phase 4 showed that strong speckle makes the detector miss ships
   silently. Estimate image quality (resolution, equivalent number of looks) before inference
   and flag low-quality chips as "not reliably surveyed" instead of "no ships".
6. **Confidence threshold.** Re-tune it on validation data for the exact deployed model and
   precision; phase 5 showed that the robust model's score calibration differs from the
   baseline's.
