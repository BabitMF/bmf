
set(OpenCV_SHARED OFF)
set(_CV_LIBS imgproc)
if(HMP_ENABLE_CUDA)
    list(APPEND _CV_LIBS cudaimgproc cudawarping)
endif()

find_package(OpenCV 4.2.0 QUIET COMPONENTS ${_CV_LIBS})