# Compiler
NVCC = nvcc

# Compiler flags
# -O3 optimizes the code for speed
# -std=c++14 sets the C++ standard
# -I tells the compiler where to find the OpenCV headers
NVCC_FLAGS = -O3 -std=c++14 -I/usr/include/opencv4

# Libraries to link (Only OpenCV is needed now!)
LIBS = -lopencv_core -lopencv_imgcodecs

# Target executable name
TARGET = defect_inspector

# Source files
SRC = main.cu

# Default rule to build the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LIBS)

# Rule to clean up compiled files
clean:
	rm -f $(TARGET)