

add_custom_target(hmp_jni_compile)

add_custom_command(TARGET hmp_jni_compile
	POST_BUILD
	    COMMAND javac -h ${CMAKE_CURRENT_BINARY_DIR} -d ${CMAKE_CURRENT_BINARY_DIR} 
	    	-cp $ENV{ANDROID_SDK_ROOT}/platforms/android-31/android.jar
                ${CMAKE_CURRENT_SOURCE_DIR}/com/bytedance/hmp/Api.java
)