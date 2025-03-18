# README
## Description
This demo illustrates the capability for multiple modules to be integrated together into a whole pipeline of preprocessing. Three are included as examples in the current `config.json`:

1. **Clip Processing**

The input video will be split into clips according to scene change, and subtitles in the video will be detected and cropped by OCR module.

2. **Aesthetic Asessment**

The video quality of each clip is assessed by the BMF provided aesthetic module in `../aesthetic_assessment`. 

3. **LLM Caption**

Each clip is then also processed by the BMF llm module in `../llm_caption` to create a description, title and summary.

The finalized video clips will be encoded as output.

## Installation

1. Pip install all the dependencies in `requirement.txt`

```
pip install -r requirement.txt
```

2. For each module used in `config.json`, please read the `README.md` for individual requirements

## Use

The entry point is `main.py`, and takes a required argument for a configuration file. 
```
python3 main.py module_config.json
```
For help, use
```
python3 main.py -h
```

An example configuration file is given in the demo - `module_config.json`.

### Configuration Schema
**Required:**

- `mode`: an ordered comma separated string of modules to use. The first module will take input, the output is passed into the next.
- `input_file`: the input path of the input video.
- `output_path`: a path to the **directory** where outputs will be written into, if the directory does not exist, one will be created. This is a **global configuration**, and will pass a prefixed path into `result_path` option for each module. If the option for that module does not exist, or is under a different name, the output for that module may be written to wherever its default location is, unless otherwise specified with an optional argument for that module.
- `output_configs`: configurations for how the final output for the module should be written if any. In the current configuration, a different resolution is specified for each file type.

**Optional**:

For each module, there can be optional additional configurations, specified with the key being the module name:

- `entry`: an entry point, take for example `aesmod_module` in the below configuration, which is needed as the module name is different to the file the module is in.
- `module_path`: if left blank, it will search in the current working directory, otherwise will search in the directory specified.
- `pre_module`: true or false, whether its a pre module.
- `options`: a dictionary of options, that will be passed in as option when running the module. These options are module specific so read the `README.md` of the specific module for options. An error will occur if a module e.g. `llm_caption` that does not produce output is chained with another.
```json
{
  "mode": "aesmod_module,ocr_crop,llm_caption",
  "input_file": "big_bunny_10s_30fps.mp4",
  "output_path": "clip_output",
  "output_configs": [
    {
      "res": "orig",
      "type": "jpg"
    },
    {
      "res": "480",
      "type": "mp4"
    }
  ],
  "ocr_crop": {
    "pre_module": true
  },
  "aesmod_module": {
    "entry": "aesmod_module.BMFAesmod",
    "module_path": "../aesthetic_assessment/",
    "pre_module": false
  },
  "llm_caption": {
    "module_path": "../llm_caption/",
    "pre_module": false,
    "options": {
      "batch_size": 6,
      "multithreading": false
    }
  }
}
```


