
NVCC = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/cuda/bin/nvcc
NFLAGS = -O2 --compiler-options '-fPIC'
NLDFLAGS = --shared


CC = gcc
CFLAGS = -fPIC -O3 # -Wall #-O0 -g -ggdb

FC = gfortran
FFLAGS = -O3#-O0 -g -Wall -ggdb


SRC_DIR = src
INC_DIR = include

BIN_DIR = bin
BUILD_DIR = build
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(BUILD_DIR))

KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

C_SRC = $(SRC_DIR)/tc_int_c.c
C_OBJ = $(BUILD_DIR)/tc_int_c.o

F_SRC = $(SRC_DIR)/gpu_module.f90
F_OBJ = $(BUILD_DIR)/gpu_module.o

MAIN_SRC = $(SRC_DIR)/tc_int_f.f90
MAIN_OBJ = $(BUILD_DIR)/tc_int_f.o

OUTPUT_BIN = tc_int
OUTPUT_LIB = tc_int_cu

TARGET = $(BIN_DIR)/$(OUTPUT_BIN)

CUDA_LIBS = -lcudart -lcublas -lstdc++
CUDA_LIBDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/cuda/lib64
CUDA_INCDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/cuda/include
CUBLAS_LIBDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/math_libs/11.8/targets/x86_64-linux/lib



TC_LIBS = $(BUILD_DIR)/lib$(OUTPUT_LIB).so


all: $(TARGET)

$(TARGET): $(TC_LIBS) $(F_OBJ) $(MAIN_OBJ)
	$(FC) $^ -o $@ $(CUDA_LIBS) -l$(OUTPUT_LIB) -L$(CUDA_LIBDIR) -L$(CUBLAS_LIBDIR) -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR)

$(TC_LIBS): $(KERNEL_OBJ) $(C_OBJ)
	$(NVCC) $(NFLAGS) $(NLDFLAGS) $^ -o $@ $(CUDA_LIBS) -L$(CUDA_LIBDIR) -L$(CUBLAS_LIBDIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR)

$(C_OBJ): $(C_SRC)
	$(CC) $(CFLAGS) -I$(CUDA_INCDIR) -c $(C_SRC) -o $(C_OBJ)

$(F_OBJ): $(F_SRC)
	$(FC) $(FFLAGS) -c $(F_SRC) -o $(F_OBJ) -J$(BUILD_DIR)

$(MAIN_OBJ): $(MAIN_SRC)
	$(FC) $(FFLAGS) -c $(MAIN_SRC) -o $(MAIN_OBJ) -J$(BUILD_DIR)

.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.so $(BUILD_DIR)/*.mod $(TARGET)


