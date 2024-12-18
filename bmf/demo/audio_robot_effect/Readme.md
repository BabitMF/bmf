
# About this Demo

This demo illustrates the fundamental workflow of a BMF Audio Module, including the following key steps:

- **Data Validation**: Ensures that input audio frames meet the required specifications for sample rate, audio layout, and data type.
- **Data Caching**: Buffers audio frames to meet the algorithm’s requirements for efficient processing.
- **Data Format Conversion**: Converts `AudioFrame` tensor lists to `ndarray` for processing.
- **Algorithm Processing**: Leverages the `librosa` library to implement robot audio effects.
- **Frame Splitting and Output**: Processes the audio buffer and splits it into frames for output.

---

## Steps Needed to Run the Demo

1. **Install the BMF Package**  
   BMF can be installed in many ways; here we use `pip`:

   ```bash
   pip3 install BabitMF
   ```

2. **Install the `librosa` Package**  
   Install the `librosa` package for audio processing:

   ```bash
   pip3 install librosa
   ```

3. **Run the Demo Script in Sync Mode**  
   We provide a sync mode that allows you to debug inside the module easily. Run the `test_robot_effect_sync_mode.py` script to test the robot effect module in sync mode:

   ```bash
   python3 test_robot_effect_sync_mode.py
   ```

   You might need to customize the input and output file paths to successfully run the demo.
