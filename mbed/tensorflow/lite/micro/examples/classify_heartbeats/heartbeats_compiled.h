// This file is generated. Do not edit.
// Generated on: 18.03.2022 12:52:03

#ifndef heartbeats_GEN_H
#define heartbeats_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus heartbeats_init();
// Returns the input tensor with the given index.
TfLiteTensor *heartbeats_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *heartbeats_output(int index);
// Runs inference for the model.
TfLiteStatus heartbeats_invoke();

// Returns the number of input tensors.
inline size_t heartbeats_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t heartbeats_outputs() {
  return 1;
}

inline void *heartbeats_input_ptr(int index) {
  return heartbeats_input(index)->data.data;
}
inline size_t heartbeats_input_size(int index) {
  return heartbeats_input(index)->bytes;
}
inline int heartbeats_input_dims_len(int index) {
  return heartbeats_input(index)->dims->data[0];
}
inline int *heartbeats_input_dims(int index) {
  return &heartbeats_input(index)->dims->data[1];
}

inline void *heartbeats_output_ptr(int index) {
  return heartbeats_output(index)->data.data;
}
inline size_t heartbeats_output_size(int index) {
  return heartbeats_output(index)->bytes;
}
inline int heartbeats_output_dims_len(int index) {
  return heartbeats_output(index)->dims->data[0];
}
inline int *heartbeats_output_dims(int index) {
  return &heartbeats_output(index)->dims->data[1];
}

#endif
