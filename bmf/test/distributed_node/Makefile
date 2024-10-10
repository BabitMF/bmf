# Define compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -g -O0 -fPIC
INCLUDES = -I/root/workspace/bmf_OSPP/output/bmf/include \
           -I.
LDFLAGS = -L/root/workspace/bmf_OSPP/output/bmf/lib 
LIBS = -lengine -lbmf_module_sdk

# Source files
LIB_SRCS_COPY = copy_module.cpp
LIB_SRCS_ASSEMBLE = assemble_module.cpp
MAIN_SRCS = cpp_demo.cpp

# Object files
LIB_OBJS_COPY = $(LIB_SRCS_COPY:.cpp=.o)
LIB_OBJS_ASSEMBLE = $(LIB_SRCS_ASSEMBLE:.cpp=.o)
MAIN_OBJS = $(MAIN_SRCS:.cpp=.o)

# Dynamic library
DYNAMIC_LIB_COPY = libcopy_module.so

DYNAMIC_LIB_ASSEMBLE = libassemble_module.so

# Default target
all: cpp_demo

# Rule to create each object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDES)

# Rule to create the dynamic library for copy module
$(DYNAMIC_LIB_COPY): $(LIB_OBJS_COPY)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

# Rule to create the dynamic library for assemble module
$(DYNAMIC_LIB_ASSEMBLE): $(LIB_OBJS_ASSEMBLE)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

# Rule to create the executable with dynamic library
cpp_demo: $(MAIN_OBJS) $(DYNAMIC_LIB_COPY) $(DYNAMIC_LIB_ASSEMBLE) 
	$(CXX) $(CXXFLAGS) -o $@ $(MAIN_OBJS) -L. -lcopy_module $(LDFLAGS) $(LIBS)

# Clean up build files
clean:
	rm -f $(LIB_OBJS_COPY) $(LIB_OBJS_ASSEMBLE) $(MAIN_OBJS) cpp_demo $(DYNAMIC_LIB_COPY) $(DYNAMIC_LIB_ASSEMBLE)

.PHONY: all clean


# # Rule to create each executable
# cpp_demo: cpp_demo.cpp
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS) $(LIBS)


# # Clean up build files
# clean:
# 	rm -f $(TARGETS) 

# .PHONY: all clean