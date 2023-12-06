# Parallel schedule of BMF, better perf than FFmpeg cmd


- This demo demonstates the better performance of BMF by parallel schedule than totally same case with FFmpeg command line.  The 1 to N video transcode case is selected as the demo.

- Steps:
  - Just need to run the script of "perf_compare.sh", it will run BMF 1 to 4 video transcode firstly, and then using the BMF tool to translate BMF json graph to be a FFmpeg command to run the same case.
  ```Bash
    ./perf_compare.sh
  ```
  - The comparison results stored into the file "compare_results.txt" like this:
  ```Text
    BMF time cost (ms): 13002.01940536499
    FFmpeg time cost (ms): 17530.67970275879
  ```




