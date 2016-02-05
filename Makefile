# http://scottmcpeak.com/autodepend/autodepend.html
# https://www.gnu.org/software/make/manual/make.html#Rules
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz3ODhDRl4y
# http://owen.sj.ca.us/~rk/howto/slides/make/slides/makerecurs.html


NVCC_PATH := /usr/local/cuda/bin
# NVCC Compiler
NVCC := $(NVCC_PATH)/nvcc
# Linker
LINK := $(NVCC_PATH)/nvcc
#
RM = rm -f

# Options for the nvcc compiler
# NVCC_FLAGS := -Xcompiler -Wall -G -g -O0 -gencode arch=compute_20,code=sm_20 -fmad=false
NVCC_FLAGS := -Xcompiler -Wall -O2 -gencode arch=compute_30,code=sm_30 -fmad=false
# NVCC_FLAGS := -Xcompiler -Wall -O2 -gencode arch=compute_20,code=sm_20 -fmad=false
# NVCC_FLAGS := -Xcompiler -Wall -O2 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -fmad=false
# NVCC_FLAGS := -Xcompiler -Wall -O2 -fmad=false

# Paths
SRC             := src
RED             := $(SRC)/red.cuda
RED_TEST        := $(SRC)/red.code_snippet.cuda.test
RED_INITIAL     := $(SRC)/red.cuda.initial
RED_UTILITY     := $(SRC)/red.cuda.utility
RED_PROCESSDATA := $(SRC)/red.cuda.processdata
RED_BENCHMARK   := $(SRC)/red.cuda.benchmark

INCLUDES        := -Ired.type -I$(RED_UTILITY)

RED_LIBRARY     := red.library
BIN             := bin

# List the objects for the executables and library
RED_OBJS := \
analytic_gas_disk.o \
fargo_gas_disk.o \
int_euler.o \
int_rungekutta2.o \
int_rungekutta4.o \
int_rungekutta8.o \
integrator.o \
main.o \
nbody_exception.o \
options.o \
parameter.o \
pp_disk.o \
test.o

RED_INITIAL_OBJS := \
distribution.o \
main.o \
nebula.o \
util_init.o

RED_UTILITY_OBJS := \
file_util.o \
red_test.o \
tokenizer.o \
tools.o \
util.o

RED_PROCESSDATA_OBJS := \
main.o

RED_BENCHMARK_OBJS := \
main.o

RED_DEPS := $(RED_OBJS:.o=.d)
RED_INITIAL_DEPS := $(RED_INITIAL_OBJS:.o=.d)
RED_UTILITY_DEPS := $(RED_UTILITY_OBJS:.o=.d)
RED_PROCESSDATA_DEPS := $(RED_PROCESSDATA_OBJS:.o=.d)
RED_BENCHMARK_DEPS := $(RED_BENCHMARK_OBJS:.o=.d)

# Targets
all : redutilcu red redinit redtest redprocess redbenchmark

red : redutilcu $(RED)/red | $(BIN)

redinit : redutilcu $(RED_INITIAL)/redinit | $(BIN)

redtest : redutilcu $(RED_TEST)/redtest | $(BIN)

redprocess : redutilcu $(RED_PROCESSDATA)/redprocess | $(BIN)

redbenchmark : redutilcu $(RED_BENCHMARK)/redbenchmark | $(BIN)

redutilcu : $(RED_UTILITY)/redutil.a | $(RED_LIBRARY)

-include $(addprefix $(RED)/, $(RED_DEPS))
-include $(addprefix $(RED_INITIAL)/, $(RED_INITIAL_DEPS))
-include $(addprefix $(RED_UTILITY)/, $(RED_UTILITY_DEPS))
-include $(addprefix $(RED_PROCESSDATA)/, $(RED_PROCESSDATA_DEPS))
-include $(addprefix $(RED_BENCHMARK)/, $(RED_BENCHMARK_DEPS))

print:
	@echo 'Red dependency files: $(RED_DEPS)'
	@echo 'Red initial dependency files: $(RED_INITIAL_DEPS)'
	@echo 'Red process dependency files: $(RED_PROCESSDATA_DEPS)'
	@echo 'Red benchmark dependency files: $(RED_BENCHMARK_DEPS)'


# Build rules
$(RED)/red : $(addprefix $(RED)/, $(RED_OBJS)) | $(BIN)
	$(LINK) $(RED_LIBRARY)/redutil.a -o $@ $?
	cp $@ $(BIN)/

$(RED_TEST)/redtest : $(RED_TEST)/main.o | $(BIN)
	$(LINK) $(RED_LIBRARY)/redutil.a -o $@ $?
	cp $@ $(BIN)/
	
$(RED_INITIAL)/redinit : $(addprefix $(RED_INITIAL)/, $(RED_INITIAL_OBJS)) | $(BIN)
	$(LINK) $(RED_LIBRARY)/redutil.a -o $@ $?
	cp $@ $(BIN)/

$(RED_PROCESSDATA)/redprocess : $(addprefix $(RED_PROCESSDATA)/, $(RED_PROCESSDATA_OBJS)) | $(BIN)
	$(LINK) $(RED_LIBRARY)/redutil.a -o $@ $?
	cp $@ $(BIN)/
	
$(RED_BENCHMARK)/redbenchmark : $(addprefix $(RED_BENCHMARK)/, $(RED_BENCHMARK_OBJS)) | $(BIN)
	$(LINK) $(RED_LIBRARY)/redutil.a -o $@ $?
	cp $@ $(BIN)/
	
$(RED_UTILITY)/redutil.a : $(addprefix $(RED_UTILITY)/, $(RED_UTILITY_OBJS)) | $(RED_LIBRARY)
	ar cr $@ $?
	cp $@ $(RED_LIBRARY)/

# compile and generate dependency info
$(RED)/%.o : $(RED)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED)/%.o : $(RED)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
#$(RED_TEST)/%.o : $(RED_TEST)/%.cpp
#	@echo 'Building file: $<'
#	@echo 'Invoking $(NVCC) Compiler'
#	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
#	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
#	@echo 'Finished building: $<'
#	@echo ''

# compile and generate dependency info
$(RED_TEST)/%.o : $(RED_TEST)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_INITIAL)/%.o : $(RED_INITIAL)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_INITIAL)/%.o : $(RED_INITIAL)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_PROCESSDATA)/%.o : $(RED_PROCESSDATA)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_PROCESSDATA)/%.o : $(RED_PROCESSDATA)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_BENCHMARK)/%.o : $(RED_BENCHMARK)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_BENCHMARK)/%.o : $(RED_BENCHMARK)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_UTILITY)/%.o : $(RED_UTILITY)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

# compile and generate dependency info
$(RED_UTILITY)/%.o : $(RED_UTILITY)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking $(NVCC) Compiler'
	$(NVCC) -c $(NVCC_FLAGS) $(INCLUDES) -o $@ $<
	$(NVCC) -M -odir "" $(NVCC_FLAGS) $(INCLUDES) -o "$(@:%.o=%.d)" $<
	@echo 'Finished building: $<'
	@echo ''

$(RED_LIBRARY) :
	mkdir $(RED_LIBRARY)

$(BIN) : 
	mkdir $(BIN)

clean:
	-$(RM) $(RED_LIBRARY)/*.a $(RED_UTILITY)/*.o $(RED_UTILITY)/*.d $(RED_INITIAL)/*.o $(RED_INITIAL)/*.d $(RED_TEST)/*.o $(RED_TEST)/*.d $(RED_PROCESSDATA)/*.o $(RED_PROCESSDATA)/*.d $(RED_BENCHMARK)/*.o $(RED_BENCHMARK)/*.d $(RED)/*.o $(RED)/*.d $(BIN)/*
