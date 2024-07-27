
NVCC = nvcc
NFLAG =

CC = gcc
CFLAG = -O3 -Wall #-O0 -g -ggdb

SRC_DIR = src
BIN_DIR = bin
$(shell mkdir -p $(BIN_DIR))

KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BIN_DIR)/%.o)

MAIN_SRC = $(SRC_DIR)/jast_der_c.c
MAIN_OBJ = $(BIN_DIR)/jast_der_c.o

TARGET = $(BIN_DIR)/jast_der_c

CUDA_LIBS = -lcudart
CUDA_LIBDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/compilers/lib
CUDA_INCDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/compilers/include


all: $(TARGET)

$(TARGET): $(KERNEL_OBJ) $(MAIN_OBJ)
	$(NVCC) $(NFLAG) $^ -o $@ $(CUDA_LIBS) -L$(CUDA_LIBDIR)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAG) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAG) -I$(CUDA_INCDIR) -c $(MAIN_SRC) -o $(MAIN_OBJ)

.PHONY: clean
clean:
	rm -f $(BIN_DIR)/*.o $(TARGET)


