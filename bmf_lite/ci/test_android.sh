#!/bin/bash
set -e
# PWD=$(dirname "$0")
# echo $PWD
# cd $PWD

device_id=$1
# log_file_base=/data/local/tmp/hydra_ci/work_space/log/
curr_date=`date +%Y-%m-%d`
curr_time=`date +%H-%M-%S`
root_work_space="/data/local/tmp/bmf_ci/work_space"
task_work_space="${root_work_space}/${curr_date}_${curr_time}"
echo ${task_work_space}
# adb -s ${device_id} shell "rm -rf ${task_work_space}/*"
adb -s ${device_id} shell "mkdir -p ${task_work_space}/"

# upload so and bin
# adb -s ${device_id} push output/arm64-v8a/bmf/libbmf_lite.so ${task_work_space}/
# adb -s ${device_id} push output/arm64-v8a/bmf/libc++_shared.so ${task_work_space}/
adb -s ${device_id} push build_android/bin/test_bmf_lite_android_interface ${task_work_space}/
adb -s ${device_id} push demo/test_data/test.jpg ${task_work_space}/
adb -s ${device_id} push demo/test_data/test-canny.png ${task_work_space}/
if [ "$(uname)" = 'Darwin' ]
then
  adb -s ${device_id} push ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/darwin-x86_64/lib64/clang/9.0.8/lib/linux/aarch64/libomp.so ${task_work_space}/
else
  adb -s ${device_id} push /usr/local/android-ndk-r21b/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/lib/linux/aarch64/libomp.so ${task_work_space}/
fi

echo "cd  ${task_work_space} &&" > adb_run_cmd.txt
echo "export LD_LIBRARY_PATH=${task_work_space}:$LD_LIBRARY_PATH &&" >> adb_run_cmd.txt
echo "export LD_LIBRARY_PATH=/data/local/tmp/control-net-execution:$LD_LIBRARY_PATH &&" >> adb_run_cmd.txt
echo './test_bmf_lite_android_interface &&' >> adb_run_cmd.txt
echo 'echo "clear workspace" &&' >> adb_run_cmd.txt
echo 'cd ../../ &&' >> adb_run_cmd.txt
# echo "rm -rf ${task_work_space} &&" >> adb_run_cmd.txt
#
echo 'echo "all test ok :)"' >> adb_run_cmd.txt

adb -s ${device_id} shell < adb_run_cmd.txt
echo 'ALL TESTES PASSED :)'
