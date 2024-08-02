
NVCC = nvcc
NFLAGS = -O2 --compiler-options '-fPIC'
NLDFLAGS = --shared


CC = gcc
CFLAGS = -fPIC -O3 # -Wall -g -ggdb

FC = gfortran
FFLAGS = -O3 # -Wall -g -ggdb


SRC_DIR = src
INC_DIR = include

BIN_DIR = bin
BUILD_DIR = build
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(BUILD_DIR))

KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

C_SRC = $(SRC_DIR)/tc_int_c.c $(SRC_DIR)/deb_int2_grad1_u12_ao.c
C_OBJ = $(BUILD_DIR)/tc_int_c.o $(BUILD_DIR)/deb_int2_grad1_u12_ao.o

F_SRC = $(SRC_DIR)/gpu_module.f90
F_OBJ = $(BUILD_DIR)/gpu_module.o

MAIN_SRC = $(SRC_DIR)/tc_int_f.f90
MAIN_OBJ = $(BUILD_DIR)/tc_int_f.o

OUTPUT_BIN = tc_int
OUTPUT_LIB = tc_int_cu

TARGET = $(BIN_DIR)/$(OUTPUT_BIN)
TARGET_DEB = $(BIN_DIR)/deb_binding

CUDA_LIBS = -lcudart -lcublas


TC_LIBS = $(BUILD_DIR)/lib$(OUTPUT_LIB).so

all: $(TARGET) 

$(TARGET): $(TC_LIBS) $(F_OBJ) $(MAIN_OBJ)
	$(FC) $^ -o $@ $(CUDA_LIBS) -l$(OUTPUT_LIB) -L$(BUILD_DIR) -I$(INC_DIR) -Wl,-rpath,$(BUILD_DIR)

$(TC_LIBS): $(KERNEL_OBJ) $(C_OBJ)
	$(NVCC) $(NFLAGS) $(NLDFLAGS) $^ -o $@ $(CUDA_LIBS) -I$(INC_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR)

$(C_OBJ): $(C_SRC)
	@for src in $(C_SRC); do \
		obj=$(BUILD_DIR)/$$(basename $${src} .c).o; \
		echo "$(CC) $(CFLAGS) -c $$src -o $$obj -I$(INC_DIR)"; \
		$(CC) $(CFLAGS) -c $$src -o $$obj -I$(INC_DIR); \
	done

$(F_OBJ): $(F_SRC)
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BUILD_DIR)

$(MAIN_OBJ): $(MAIN_SRC)
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BUILD_DIR)




$(TARGET_DEB): $(BUILD_DIR)/libdebbind.so $(BUILD_DIR)/deb_module.o $(BUILD_DIR)/deb_binding.o
	$(FC) $^ -o $@ $(CUDA_LIBS) -ldebbind -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR)

$(BUILD_DIR)/libdebbind.so: $(BUILD_DIR)/deb_fct_c.o
	$(NVCC) $(NFLAGS) $(NLDFLAGS) $^ -o $@ $(CUDA_LIBS)

$(BUILD_DIR)/deb_fct_c.o: $(SRC_DIR)/deb_fct_c.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/deb_module.o: $(SRC_DIR)/deb_module.f90
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BUILD_DIR)

$(BUILD_DIR)/deb_binding.o: $(SRC_DIR)/deb_binding.f90
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BUILD_DIR)




.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.so $(BUILD_DIR)/*.mod $(BIN_DIR)/*


