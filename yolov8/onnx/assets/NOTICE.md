# ONNX export sample assets

The image files in this directory are bundled so that the ONNX consistency
checker (``yolov8.onnx.consistency.compare_onnx_vs_torch``) has a real
photograph to feed both PyTorch and onnxruntime through. They are *not*
training data, just a fixed reference input for numerical-parity tests.

## Provenance

* ``sample_bus.jpg`` — copied verbatim from
  ``ultralytics/assets/bus.jpg`` (https://github.com/ultralytics/assets),
  the canonical YOLO test image redistributed across hundreds of YOLO
  derivatives. The Ultralytics assets repository publishes these test
  images as freely redistributable test fixtures.

If a downstream consumer prefers a different image, ``compare_onnx_vs_torch``
accepts an explicit ``image=`` argument; this file is only the default.
