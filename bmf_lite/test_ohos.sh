#!/bin/bash
set -e

device_id=$1
curr_date=`date +%Y-%m-%d`
curr_time=`date +%H-%M-%S`
root_work_space="/data/local/tmp/bmf_ci/work_space"
task_work_space="${root_work_space}/${curr_date}_${curr_time}"
echo ${task_work_space}
hdc -t ${device_id} shell "rm -rf ${task_work_space}/*"
hdc -t ${device_id} shell "mkdir -p ${task_work_space}/"

# upload so and bin
hdc -t ${device_id} file send build_ohos/bin/test_bmf_lite_ohos_interface ${task_work_space}/
hdc -t ${device_id} file send demo/test_data/ ${task_work_space}/
if [ "$(uname)" = 'Darwin' ]
then
    hdc -t ${device_id} file send ${OHOS_NATIVE_ROOT}/sysroot/usr/lib/aarch64-linux-ohos/libEGL.so ${task_work_space}/
    hdc -t ${device_id} file send ${OHOS_NATIVE_ROOT}/sysroot/usr/lib/aarch64-linux-ohos/libGLESv3.so ${task_work_space}/
    # hdc -t ${device_id} file send ${OHOS_NATIVE_ROOT}/sysroot/usr/lib/aarch64-linux-ohos/libhilog_ndk.z.so ${task_work_space}/
fi
echo "cd  ${task_work_space} &&" > adb_run_cmd.txt
echo "export LD_LIBRARY_PATH=${task_work_space}:$LD_LIBRARY_PATH &&" >> adb_run_cmd.txt


echo './test_bmf_lite_ohos_interface &&' >> adb_run_cmd.txt

echo 'echo "clear workspace" &&' >> adb_run_cmd.txt
echo 'cd ../../ &&' >> adb_run_cmd.txt
echo "rm -rf ${task_work_space} &&" >> adb_run_cmd.txt
#
echo 'echo "all test ok :)"' >> adb_run_cmd.txt

hdc -t ${device_id} shell < adb_run_cmd.txt
echo 'ALL TESTES PASSED :)'

# 我好不容易写完了才发现它居然不能运行
