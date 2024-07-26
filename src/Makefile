
NVCC = nvcc
NFLAG =

CC = gcc
CFLAG = -O3#-O0 -g -Wall -ggdb

OBJ_DIR = obj
$(shell mkdir -p $(OBJ_DIR))

KERNEL_SRC = utils.cu jast_bh.cu
KERNEL_OBJ = $(OBJ_DIR)/utils.o $(OBJ_DIR)/jast_bh.o

MAIN_SRC = jast_der_c.c
MAIN_OBJ = $(OBJ_DIR)/jast_der_c.o

OUTPUT = jast_der_c

CUDA_LIBS = -lcudart
CUDA_LIBDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/compilers/lib
CUDA_INCDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/23.9/compilers/include


TARGET: $(OUTPUT)

$(OUTPUT): $(KERNEL_OBJ) $(MAIN_OBJ)
	$(NVCC) $(MAIN_OBJ) $(KERNEL_OBJ) -o $(OUTPUT) $(CUDA_LIBS) -L$(CUDA_LIBDIR)

$(OBJ_DIR)/%.o: %.cu
	$(NVCC) $(NFLAG) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAG) -I$(CUDA_INCDIR) -c $(MAIN_SRC) -o $(MAIN_OBJ)

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o $(OUTPUT)


