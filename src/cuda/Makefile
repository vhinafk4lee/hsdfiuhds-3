# RTX 5090 is Blackwell -> sm_120. Override for other cards, e.g. CUDA_ARCH=89 (4090).
CUDA_ARCH ?= 120
NVCC      ?= nvcc
TARGET    := hcminer-gpu

GENCODE := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
           -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)

NVCCFLAGS := -O3 -std=c++17 --use_fast_math -Xptxas -O3,-v $(GENCODE)

all: $(TARGET)

$(TARGET): miner.cu ../keccak/keccak_core.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ miner.cu

clean:
	rm -f $(TARGET)

.PHONY: all clean
