# Mission 4: Can It Run on the ESP32-S3?

**Project:** EDGE AI – Team 13  
**Project Title:** How Long Will the Battery Last?  
**Model:** Battery Life Prediction Neural Network  
**Mission:** Mission 4 – Edge Deployment and Memory Analysis

---

## 1. Introduction

The objective of Mission 4 is to evaluate whether the trained battery-life prediction neural network is suitable for deployment on an ESP32-S3 edge device.

The analysis focuses on:

- Model parameter count and storage requirements.
- Float32 and theoretical INT8 model sizes.
- SRAM and PSRAM memory budgets.
- Activation memory requirements.
- Multiply-accumulate (MAC) operations.
- Model-width comparisons.
- A theoretical inference-time estimate.
- The trade-offs between cloud-based and local edge inference.

The analysis uses the trained model and calculated memory requirements. An actual ESP32-S3 deployment and hardware inference benchmark were not performed.

## 2. ESP32-S3 Hardware Overview

The ESP32-S3 is a dual-core 32-bit Xtensa LX7 microcontroller from Espressif.

According to the official Espressif documentation, its relevant specifications include:

| Hardware property      | Specification                  |
| ---------------------- | ------------------------------ |
| Processor              | Dual-core Xtensa LX7           |
| Maximum CPU frequency  | Up to 240 MHz                  |
| On-chip SRAM           | 512 KB                         |
| RTC SRAM               | 16 KB                          |
| Floating-point support | Single-precision FPU           |
| Wireless connectivity  | Wi-Fi 2.4 GHz and Bluetooth LE |

The assignment's assumed deployment memory budgets are:

| Memory resource                          |          Assumed budget |
| ---------------------------------------- | ----------------------: |
| Available SRAM budget for this analysis  | 300 KB (307,200 bytes)  |
| PSRAM budget                             | 8 MB (8,388,608 bytes)  |
| Flash capacity                           |                   16 MB |

**Important:** The 300 KB SRAM figure is the available application budget assumed for this analysis, not the total physical SRAM of the chip. The actual free memory depends on firmware, runtime, peripherals and operating configuration. PSRAM and flash capacities depend on the selected ESP32-S3 module or board.

Official references:

- ESP32-S3 Series Datasheet: https://documentation.espressif.com/esp32_s3_datasheet_en.pdf
- ESP32-S3 product page: https://www.espressif.com/en/products/socs/esp32-s3
- ESP-IDF power management documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/system/power_management.html

## 3. Actual Trained Model Architecture

The trained battery-life prediction model has the following architecture:

**5 → 16 → 16 → 1**

| Property              |     Value |
| --------------------- | --------: |
| Input features        |         5 |
| Hidden layers         |         2 |
| Hidden units          | 16 and 16 |
| Output units          |         1 |
| Total parameters      |       385 |
| Validation MAE        | 6.42 days |
| Total MAC operations  |       352 |

The five input features are:

1. Battery voltage (`battery_v`)
2. Temperature (`temp_c`)
3. Rings per day (`rings_per_day`)
4. Wi-Fi signal strength (`wifi_rssi_dbm`)
5. Battery age (`age_months`)

### 3.1 Parameter calculation

The model consists of three fully connected layers.

| Layer     | Dimensions | Weights | Biases | Parameters |
| --------- | ---------- | ------: | -----: | ---------: |
| Linear 1  | 5 → 16     |      80 |     16 |         96 |
| Linear 2  | 16 → 16    |     256 |     16 |        272 |
| Linear 3  | 16 → 1     |      16 |      1 |         17 |
| **Total** |            | **352** | **33** |    **385** |

Therefore:

`Total parameters = 352 weights + 33 biases = 385`

### 3.2 MAC calculation

For fully connected layers, the number of MAC operations is calculated by multiplying the input width by the output width.

- Linear 1: 5 × 16 = 80 MACs
- Linear 2: 16 × 16 = 256 MACs
- Linear 3: 16 × 1 = 16 MACs

**Total MAC operations = 80 + 256 + 16 = 352 MACs per prediction.**

This count describes the multiply-accumulate operations in the linear layers. It does not include activation functions or other software overhead.

## 4. Model Memory Requirements

The model's parameter storage was calculated for Float32 and theoretical INT8 representations.

### 4.1 Actual model storage

| Measurement                     |     Float32 | Theoretical INT8 |
| ------------------------------- | ----------: | ---------------: |
| Parameters                      |         385 |              385 |
| Bytes per parameter             |           4 |                1 |
| Model parameter storage         | 1,540 bytes |        385 bytes |
| Fits within 300 KB SRAM budget  |         Yes |              Yes |
| Fits within 8 MB PSRAM budget   |         Yes |              Yes |

Float32 storage:

`385 × 4 = 1,540 bytes`

Theoretical INT8 storage:

`385 × 1 = 385 bytes`

The theoretical INT8 representation reduces parameter storage by 75% compared with Float32.

**Important:** INT8 storage is a theoretical estimate. The project has not performed actual model quantisation, calibration or accuracy testing. Therefore, no claim is made that an INT8 model has been successfully deployed or that its prediction accuracy has been preserved.

### 4.2 Layer-by-layer memory

| Layer     | Parameters | Float32 storage |
| --------- | ---------: | --------------: |
| Linear 1  |         96 |       384 bytes |
| Linear 2  |        272 |     1,088 bytes |
| Linear 3  |         17 |        68 bytes |
| **Total** |    **385** | **1,540 bytes** |

## 5. Activation Memory Estimate

Model weights are not the only memory required during inference. Intermediate layer outputs, also called activations, must also be stored.

For the actual architecture:

`5 → 16 → 16 → 1`

The largest hidden-layer output contains 16 values.

For Float32 activations:

`16 × 4 = 64 bytes`

Therefore, the largest single hidden-layer activation is estimated to require **64 bytes**.

This is an estimate of the largest individual hidden-layer output buffer, not the total runtime RAM requirement.

The complete application will also require memory for:

- Input and output tensors.
- Temporary inference buffers.
- Runtime and framework data.
- Stack and heap allocations.
- Wi-Fi and other application services.
- Firmware and peripheral operation.

Consequently, the 64-byte activation estimate should not be interpreted as the complete RAM requirement of the deployed application.

## 6. Model-Width Comparison

To evaluate how model size changes with architecture width, the same network structure was considered with different hidden widths:

`5 → w → w → 1`

The parameter formula is:

`Parameters = w² + 8w + 1`

The MAC formula is:

`MACs = w² + 6w`

Float32 storage is:

`Parameters × 4 bytes`

Theoretical INT8 storage is:

`Parameters × 1 byte`

The largest single hidden-layer activation estimate is:

`w × 4 bytes`

### 6.1 Comparison results

| Hidden width | Parameters |   Float32 size |    INT8 size | Largest activation |      MACs |
| -----------: | ---------: | -------------: | -----------: | -----------------: | --------: |
|           32 |      1,281 |        5,124 B |      1,281 B |              128 B |     1,216 |
|          128 |     17,409 |       69,636 B |     17,409 B |              512 B |    17,152 |
|          256 |     67,585 |      270,340 B |     67,585 B |            1,024 B |    67,072 |
|          512 |    266,241 |    1,064,964 B |    266,241 B |            2,048 B |   265,472 |
|         2048 |  4,210,689 |   16,842,756 B |  4,210,689 B |            8,192 B | 4,202,624 |

### 6.2 SRAM and PSRAM fit comparison

The following results compare parameter storage alone against the assumed memory budgets.

| Hidden width | Float32 fits 300 KB SRAM | INT8 fits 300 KB SRAM | Float32 fits 8 MB PSRAM | INT8 fits 8 MB PSRAM |
| -----------: | ------------------------ | --------------------- | ----------------------- | -------------------- |
|           32 | Yes                      | Yes                   | Yes                     | Yes                  |
|          128 | Yes                      | Yes                   | Yes                     | Yes                  |
|          256 | Yes                      | Yes                   | Yes                     | Yes                  |
|          512 | No                       | Yes                   | Yes                     | Yes                  |
|         2048 | No                       | No                    | No                      | Yes                  |

These fit results consider model weights and biases only. They do not establish that the complete application will fit within the available memory.

### 6.3 Maximum hidden width under the SRAM budget

For the architecture `5 → w → w → 1`, the parameter count is:

`w² + 8w + 1`

Using the 307,200-byte SRAM budget:

**Float32:**

- Maximum width: 273
- Storage at width 273: 306,772 bytes
- Width 274 exceeds the budget at 309,024 bytes.

**Theoretical INT8:**

- Maximum width: 550
- Storage at width 550: 306,901 bytes
- Width 551 exceeds the budget at 308,004 bytes.

This comparison shows how theoretical INT8 storage can permit wider models within the same parameter-storage budget. It does not prove that such wider models would achieve acceptable battery-life prediction accuracy.

## 7. Theoretical Inference-Time Estimate

The ESP32-S3 supports CPU frequencies up to 240 MHz.

The actual model requires 352 MAC operations per prediction in its fully connected layers.

A simple operation-throughput illustration can be calculated using:

`Estimated time = (MAC count × assumed cycles per MAC) / CPU frequency`

At 240 MHz:

| Assumed cycles per MAC | Illustrative arithmetic time |
| ---------------------: | ---------------------------: |
|                      1 |            1.47 microseconds |
|                     10 |           14.67 microseconds |
|                    100 |          146.67 microseconds |

For example, assuming one cycle per MAC:

`352 / 240,000,000 seconds ≈ 1.47 microseconds`

**These figures are illustrative calculations, not measured ESP32-S3 inference times.** The cycles-per-MAC assumptions have not been benchmarked on the target hardware. Actual inference time will depend on the implementation, compiler, memory access, activation functions, CPU frequency, runtime overhead and other system activity.

The 1.47-microsecond figure must not be presented as a verified hardware result or guaranteed lower bound.

A real inference-time measurement would require deploying the model to a specific ESP32-S3 board and timing the prediction using the chosen inference implementation.

## 8. Cloud vs Edge Deployment Comparison

The model can be used through a remote prediction API or potentially deployed locally on an edge device.

| Factor               | Cloud/server inference                                | ESP32-S3 edge inference                                  |
| -------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| Latency              | Includes network communication and server response    | Can avoid network round-trip delay for local prediction  |
| Internet connection  | Required for remote API access                        | Not required for local prediction after deployment        |
| Privacy              | Sensor readings may be sent to a remote server        | Readings can remain on the device                        |
| Cost                 | May involve server hosting and network costs          | Requires device hardware and maintenance                 |
| Power                | Device uses communication hardware to send readings   | Local inference uses device processing resources         |
| Availability         | Depends on server and network availability            | Local prediction can continue without Wi-Fi              |
| Model updates        | Central updates can be easier to manage               | Device model or firmware updates are required            |
| Processing resources | Can use larger server resources                       | Limited by device memory and processor capacity          |

### 8.1 Latency

The Mission 3 laptop-to-Codespace experiment measured a median round-trip time of 34.024 ms, while the server processing time was 0.159 ms.

This demonstrated that a significant part of remote API response time occurred outside the measured server processing stages.

An edge implementation could avoid this network round trip for predictions performed entirely on the device.

However, actual ESP32-S3 inference latency has not yet been measured.

### 8.2 Privacy

Local inference can keep sensor readings on the smart doorbell instead of transmitting every reading to a remote prediction service.

This can reduce the amount of sensor data sent externally. Other device communications may still transmit information depending on the application design.

### 8.3 Cost and power

Cloud inference may require ongoing server hosting and network communication.

Edge inference can reduce dependence on a remote prediction service, but the hardware, firmware development, maintenance and device power consumption must also be considered.

Local computation does not automatically guarantee lower total energy consumption. This would need to be measured on the target hardware.

### 8.4 Wi-Fi availability

A local prediction model can continue to calculate battery-life estimates when Wi-Fi is unavailable, provided that the model and application are already installed on the device.

A remote API-based prediction system cannot obtain a new remote prediction while its network connection is unavailable.

## 9. Deployment Assessment

The actual trained model contains only 385 parameters and requires 1,540 bytes of Float32 parameter storage.

Its largest single hidden-layer Float32 activation estimate is 64 bytes, and its fully connected layers require 352 MAC operations per prediction.

These requirements are small compared with the assumed 300 KB SRAM parameter-storage budget.

Based on the parameter-storage and activation estimates, the actual model appears suitable for further ESP32-S3 deployment investigation.

However, complete deployment feasibility cannot be confirmed from model parameter storage alone. The runtime, firmware, temporary buffers, networking components and available heap must also be considered.

A real board-level implementation and measurement would be needed to confirm final memory usage, inference latency, prediction accuracy and power consumption.

## 10. Limitations and Assumptions

The following limitations apply to this analysis:

1. No actual ESP32-S3 hardware inference benchmark was performed.
2. Inference-time values are illustrative calculations based on assumed cycles per MAC.
3. The INT8 model size is theoretical; no quantised model has been generated or accuracy-tested.
4. The 300 KB SRAM budget is an assignment assumption, not the total physical SRAM of the ESP32-S3.
5. The 8 MB PSRAM and 16 MB flash figures are assumed deployment capacities; actual capacities depend on the selected board or module.
6. Activation memory represents the largest single hidden-layer output, not total runtime RAM.
7. Memory fit calculations compare model parameters alone against the stated memory budgets.
8. Actual power consumption and energy per prediction have not been measured.

## 11. Conclusion

Mission 4 evaluated the memory and computational requirements of the battery-life prediction model for a possible ESP32-S3 edge deployment.

The actual trained network has 385 parameters, requires 1,540 bytes of Float32 parameter storage and performs 352 MAC operations per prediction.

The largest single hidden-layer activation is estimated at 64 bytes using Float32 values.

The model-width comparison showed that larger architectures can eventually exceed the assumed SRAM budget, while theoretical INT8 storage can reduce parameter memory requirements.

The theoretical inference-time calculations illustrate the relationship between operation count, CPU frequency and assumed processing cycles. They are not a substitute for a real hardware benchmark.

Overall, the actual trained model has a small parameter and activation footprint and appears promising for further edge deployment testing. A real ESP32-S3 implementation is still required to verify inference time, complete memory usage, prediction accuracy and power consumption.

## 12. References

1. Espressif Systems, ESP32-S3 Series Datasheet: https://documentation.espressif.com/esp32_s3_datasheet_en.pdf
2. Espressif Systems, ESP32-S3 Product Page: https://www.espressif.com/en/products/socs/esp32-s3
3. Espressif Systems, ESP-IDF Power Management Documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/system/power_management.html

---

**Mission 4 status:** Memory calculations, model-width comparisons and `/model_info` implementation were completed. This report documents the theoretical edge deployment assessment. Actual ESP32-S3 hardware deployment and benchmarking remain future validation work.
