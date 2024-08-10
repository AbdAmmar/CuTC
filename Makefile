
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
BLD_DIR = build
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(BLD_DIR))


CU_SRC = $(wildcard $(SRC_DIR)/*.cu)
CU_OBJ = $(CU_SRC:$(SRC_DIR)/%.cu=$(BLD_DIR)/%.o)

C_SRC = $(SRC_DIR)/cutc_int_c.c $(SRC_DIR)/deb_int_2e_ao.c $(SRC_DIR)/cutc_no_2e.c $(SRC_DIR)/deb_no_2e.c $(SRC_DIR)/cutc_no_1e.c $(SRC_DIR)/deb_no_1e.c
C_OBJ = $(BLD_DIR)/cutc_int_c.o $(BLD_DIR)/deb_int_2e_ao.o $(BLD_DIR)/cutc_no_2e.o $(BLD_DIR)/deb_no_2e.o $(BLD_DIR)/cutc_no_1e.o $(BLD_DIR)/deb_no_1e.o


F_SRC = $(SRC_DIR)/cutc_module.f90
F_OBJ = $(BLD_DIR)/cutc_module.o

MAIN_SRC = $(SRC_DIR)/cutc_int_f.f90
MAIN_OBJ = $(BLD_DIR)/cutc_int_f.o

OUTPUT_LIB = cutcint

TARGET = $(BIN_DIR)/cutc_int



CUDA_LIBS = -lcudart -lcublas


TC_LIBS = $(BLD_DIR)/lib$(OUTPUT_LIB).so

all: $(TARGET)

$(TARGET): $(TC_LIBS) $(F_OBJ) $(MAIN_OBJ)
	$(FC) $^ -o $@ $(CUDA_LIBS) -l$(OUTPUT_LIB) -L$(BLD_DIR) -I$(INC_DIR) -Wl,-rpath,$(BLD_DIR)

$(TC_LIBS): $(CU_OBJ) $(C_OBJ)
	$(NVCC) $(NFLAGS) $(NLDFLAGS) $^ -o $@ $(CUDA_LIBS) -I$(INC_DIR)

$(BLD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR)

$(C_OBJ): $(C_SRC)
	@for src in $(C_SRC); do \
		obj=$(BLD_DIR)/$$(basename $${src} .c).o; \
		echo "$(CC) $(CFLAGS) -c $$src -o $$obj -I$(INC_DIR)"; \
		$(CC) $(CFLAGS) -c $$src -o $$obj -I$(INC_DIR); \
	done

$(F_OBJ): $(F_SRC)
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BLD_DIR)

$(MAIN_OBJ): $(MAIN_SRC)
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BLD_DIR)





.PHONY: clean
clean:
	rm -f $(BLD_DIR)/*.o $(BLD_DIR)/*.so $(BLD_DIR)/*.mod $(BIN_DIR)/*


