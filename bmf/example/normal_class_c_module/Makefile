CC = g++
AT = @
ECHO = echo

ARCH=$(shell uname -s)
ifeq ($(ARCH), Darwin)
    ifeq ($(CUSTOM_BMF_PYTHON_VERSION), python3)
        INCPATHS = -I/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m
	    PYTHON_LIB = -L/usr/local/lib -L/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib
	    COMMON_LIBS = -lpython3.7
	    COMMON_LIBS += -lboost_python37 -lboost_numpy37 $(PYTHON_LIB)
    else
        INCPATHS = -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
        COMMON_LIBS = -lpython2.7
        COMMON_LIBS += -lboost_python27 -lboost_numpy27
    endif
else
    ifeq ($(CUSTOM_BMF_PYTHON_VERSION), python3)
        COMMON_LIBS = -lpython3.5
        COMMON_LIBS += -lboost_numpy35 -lboost_python35
        PYTHON_INC = -I$(ROOT_DIR)/3rd_party/python35/include
        PYTHON_LIB = -L$(ROOT_DIR)/3rd_party/python35/lib
    else
        COMMON_LIBS = -lpython2.7
        COMMON_LIBS += -lboost_numpy27 -lboost_python27
        PYTHON_INC = -I$(ROOT_DIR)/3rd_party/python27/include
        PYTHON_LIB = -L$(ROOT_DIR)/3rd_party/python27/lib
    endif
    BOOST_PYTHON_INC = -I$(ROOT_DIR)/3rd_party/boost-python/include
    BOOST_PYTHON_LIB = -L$(ROOT_DIR)/3rd_party/boost-python/lib

    INCPATHS = $(BOOST_PYTHON_INC) $(PYTHON_INC)
    COMMON_LIBS += $(BOOST_PYTHON_LIB) $(PYTHON_LIB)
endif

ROOT_DIR = ../../..

BMF_LIB = $(ROOT_DIR)/bmf/c_module_sdk/build/bin/lib/libbmf_sdk.a
BMF_INC = -I$(ROOT_DIR)/bmf/c_module_sdk/build/bin/include

COMMON_LIBS += $(BMF_LIB)

COMMON_FLAGS = -fPIC -Wno-deprecated-declarations

LD_FLAGS = -shared

INCPATHS += $(BMF_INC)

TARGET = normal_image_info.so

CC_SRC = normal_image_info.cc
CC_OBJ = $(patsubst %.cc, %.o, $(CC_SRC))


# --------------------- all ---------------------
all: $(TARGET)

# --------------------- link ---------------------
$(TARGET): $(CC_OBJ)
	$(CC) -o $@ $^ $(COMMON_LIBS) $(COMMON_FLAGS) $(LD_FLAGS)

# --------------------- compile ---------------------
%.o: %.cc
	$(CC) -c -o $@ $< $(COMMON_FLAGS) $(INCPATHS)

# --------------------- clean ---------------------
clean:
	$(AT)rm -fr $(CC_OBJ) $(TARGET)
