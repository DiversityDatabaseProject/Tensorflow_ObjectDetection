'''
Converts base model to tflite with dynamic quantization option (default)
Post-training quantization is a conversion technique that can reduce model size while also improving CPU and hardware accelerator latency, with little degradation in model accuracy.

Specify ops in case of below error: 
File "C:\python_versions\python_venvs\dl_py_310\lib\site-packages\tensorflow\lite\python\convert.py", line 306, in convert raise converter_error
tensorflow.lite.python.convert_phase.ConverterError: <unknown>:0: error: loc(callsite(callsite(fused["ConcatV2:", "Postprocessor/BatchMultiClassNonMaxSuppression/MultiClassNonMaxSuppression/ChangeCoordinateFrame/Scale/concat@__inference_call_func_11386"] at fused["StatefulPartitionedCall:", "StatefulPartitionedCall@__inference_signature_wrapper_13760"]) at fused["StatefulPartitionedCall:", "StatefulPartitionedCall"])): 'tf.ConcatV2' op is neither a custom op nor a flex op
'''

import  tensorflow  as  tf
import load_configs as cf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(cf.SAVED_MODEL_DIR) # path to the SavedModel directory

# Specify ops, if needed
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

#To quantize the model on export, set the optimizations flag to optimize for size:
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant = converter.convert()
# Save the model.
with  open(cf.TFLITE_MODEL_QUANTIZED, 'wb') as  f:
    f.write(tflite_model_quant)