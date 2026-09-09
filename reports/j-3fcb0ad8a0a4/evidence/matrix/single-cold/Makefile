
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -std=c++20 -mllvm -amdgpu-function-calls=false -DUSE_ROCM -fno-offload-uniform-block -U__HIP_NO_HALF_OPERATORS__ -DENABLE_CK=1 -O3 -Wno-unused-result -Wno-switch-bool --offload-arch=gfx942 -Wno-undefined-func-template -D__HIP_PLATFORM_HCC__=1 -DLEGACY_HIPBLAS_DIRECT -mllvm --amdgpu-kernarg-preload-count=16 -DUSE_PROF_API=1 -mllvm -enable-post-misched=0 -DENABLE_FP8 -Wno-vla-cxx-extension -U__HIP_NO_HALF_CONVERSIONS__ -mllvm --lsr-drop-solution=1 -fgpu-flush-denormals-to-zero -mllvm -amdgpu-early-inline-all=true -D__HIP_PLATFORM_AMD__=1 -I/job/aiter-cache-j-3fcb0ad8a0a4/single-cold/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)