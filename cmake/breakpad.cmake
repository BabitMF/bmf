if(NOT DEFINED BMF_ENABLE_BREAKPAD)
    return()
endif()

if(BREAKPAD_FOUND)
    return()
endif()

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_LOWER)

find_path(BREAKPAD_INCLUDE_DIR client/${SYSTEM_NAME_LOWER}/handler/exception_handler.h
        HINTS ${CMAKE_SOURCE_DIR}/3rd_party
        PATH_SUFFIXES breakpad/include/breakpad)
find_library(BREAKPAD_LIBRARY breakpad_client
	HINTS ${CMAKE_SOURCE_DIR}/3rd_party
	PATH_SUFFIXES breakpad/lib)

if(BREAKPAD_INCLUDE_DIR AND BREAKPAD_LIBRARY)
	message(STATUS "find BREAKPAD include dir at:" ${BREAKPAD_INCLUDE_DIR} ", BREAKPAD library dir at:" ${BREAKPAD_LIBRARY})
else()
    set(BMF_ENABLE_BREAKPAD OFF) 
    message(WARNING "can not find breakpad, disable it...")
endif()
