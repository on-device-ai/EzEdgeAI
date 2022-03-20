/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/classify_heartbeats/main_functions.h"

#define TFLITE_MICRO_COMPILED

#ifdef TFLITE_MICRO_COMPILED

  #include "tensorflow/lite/c/common.h"
  #include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
  #include "heartbeats_compiled.h"
  #include "tensorflow/lite/micro/examples/classify_heartbeats/heartbeats_signal.h"
  
  #include "tensorflow/lite/micro/micro_error_reporter.h"
  
#else
	
  #include "tensorflow/lite/micro/all_ops_resolver.h"
  #include "tensorflow/lite/micro/examples/classify_heartbeats/classify_heartbeats_cnn_quantized.h"
  #include "tensorflow/lite/micro/examples/classify_heartbeats/heartbeats_signal.h"
  #include "tensorflow/lite/micro/micro_error_reporter.h"
  #include "tensorflow/lite/micro/micro_interpreter.h"
  #include "tensorflow/lite/schema/schema_generated.h"
  #include "tensorflow/lite/version.h"
  
#endif


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
#ifndef TFLITE_MICRO_COMPILED
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // Finding the minimum value for your model may require some trial and error.
  constexpr int kTensorArenaSize = 100 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
#endif
uint8_t signal_count = 0;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  error_reporter->Report("setup() : Start");

#ifdef TFLITE_MICRO_COMPILED

  heartbeats_init();

#else
       
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(classify_heartbeats_cnn_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;  

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
  
#endif  

  signal_count = 0;

  error_reporter->Report("setup() : End");

}

#include "LCD_DISCO_F746NG.h"

LCD_DISCO_F746NG lcd;

const uint32_t background_color = 0xFFF4B400;  // Yellow
const uint32_t foreground_color = 0xFFDB4437;  // Red

bool    initialized = false;
float   loop_signal[260];
int     width;
int     height;
int     midpoint;
int     prev_x_pos;
int     prev_y_pos;
uint8_t message_str[80];

// The name of this function is important for Arduino compatibility.
void loop() {

  if (!initialized) {

    lcd.Clear(background_color);
    lcd.SetTextColor(foreground_color);

    width = lcd.GetXSize();
    height = lcd.GetYSize();

    error_reporter->Report("width = %d height = %d",width,height);

    midpoint = height / 2;

    initialized = true;
  }

  switch(signal_count) {
  case 0:
    for(int i=0;i<260;i++) {
      loop_signal[i] = signalN[i];
    }
    break;
  case 1:
    for(int i=0;i<260;i++) {
      loop_signal[i] = signalS[i];
    }
    break;
  case 2:
    for(int i=0;i<260;i++) {
      loop_signal[i] = signalV[i];
    }
    break;
  case 3:
    for(int i=0;i<260;i++) {
      loop_signal[i] = signalF[i];
    }
    break;  
  case 4:
    for(int i=0;i<260;i++) {
      loop_signal[i] = signalQ[i];
    }
    break;
  }

  lcd.Clear(background_color);
  prev_x_pos = prev_y_pos = 0;
  for(int x_value=0;x_value<240;x_value++) {
     float y_value = loop_signal[x_value];
     int   x_pos   = x_value * 2;
     int   y_pos;
     y_value = y_value / 6.0;
     if (y_value >= 0) {
       y_pos = static_cast<int>(midpoint * (1.f - y_value));
     } else {
       y_pos = midpoint + static_cast<int>(midpoint * (0.f - y_value));
     }
     lcd.DrawLine(prev_x_pos,prev_y_pos,x_pos,y_pos);
     prev_x_pos = x_pos;
     prev_y_pos = y_pos;
  }
  wait_us(1000*1000);

#ifdef TFLITE_MICRO_COMPILED

  for(int i=0;i<260;i++) {
    tflite::GetTensorData<float>(heartbeats_input(0))[i] = loop_signal[i];
  }

  heartbeats_invoke();
 
  int argmax_index = -1;
  for(int i = 0 ; i < 5 ; i++) {
    for(int j = 0 ; j < 5 ; j++) {
       if(i != j) {
         if( tflite::GetTensorData<float>(heartbeats_output(0))[i] > tflite::GetTensorData<float>(heartbeats_output(0))[j] ) {
	   argmax_index = i;
         } else {
           argmax_index = -1;
           break;
         }
       }
    }
    if(argmax_index != -1) {
       break;
    }
  }
  
  error_reporter->Report("Label is %d , prediction is %d (%f %f %f %f %f)",signal_count,argmax_index,
    tflite::GetTensorData<float>(heartbeats_output(0))[0],
	tflite::GetTensorData<float>(heartbeats_output(0))[1],
	tflite::GetTensorData<float>(heartbeats_output(0))[2],
	tflite::GetTensorData<float>(heartbeats_output(0))[3],
	tflite::GetTensorData<float>(heartbeats_output(0))[4]);
  sprintf((char *)message_str,"Label=%d,prediction=%d\0",signal_count,argmax_index);
  lcd.DisplayStringAt(0,0,message_str,LEFT_MODE);

#else
	
  for(int i=0;i<260;i++) {
    input->data.f[i] = loop_signal[i];
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed !\n");
    return;
  }
  
  int argmax_index = -1;
  for(int i = 0 ; i < 5 ; i++) {
    for(int j = 0 ; j < 5 ; j++) {
       if(i != j) {
         if(output->data.f[i] > output->data.f[j]) {
	   argmax_index = i;
         } else {
           argmax_index = -1;
           break;
         }
       }
    }
    if(argmax_index != -1) {
       break;
    }
  }
  
  error_reporter->Report("Label is %d , prediction is %d (%f %f %f %f %f)",signal_count,argmax_index,
  output->data.f[0],output->data.f[1],output->data.f[2],output->data.f[3],output->data.f[4]);
  sprintf((char *)message_str,"Label=%d,prediction=%d\0",signal_count,argmax_index);
  lcd.DisplayStringAt(0,0,message_str,LEFT_MODE);
  
#endif

  if(signal_count < 4) {
    signal_count ++;
    wait_us(2*1000*1000);
  } else {
    signal_count = 0;
    wait_us(10*1000*1000);
  }

}
