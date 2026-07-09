# Compiler
NVCC = nvcc

# Compiler flags
# -O3 optimizes the code for speed
# -std=c++14 sets the C++ standard
NVCC_FLAGS = -O3 -std=c++14

# Libraries to link (NVIDIA Performance Primitives for image processing)
LIBS = -lnppig -lnppc -lnppi

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