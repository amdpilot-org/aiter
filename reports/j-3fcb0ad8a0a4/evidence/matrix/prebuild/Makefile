
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -mllvm --amdgpu-kernarg-preload-count=16 -mllvm -amdgpu-early-inline-all=true --offload-arch=gfx942 -DUSE_PROF_API=1 -fno-offload-uniform-block -mllvm -enable-post-misched=0 -Wno-unused-result -Wno-vla-cxx-extension -O3 -mllvm --lsr-drop-solution=1 -std=c++20 -mllvm -amdgpu-function-calls=false -U__HIP_NO_HALF_OPERATORS__ -DENABLE_FP8 -U__HIP_NO_HALF_CONVERSIONS__ -fgpu-flush-denormals-to-zero -Wno-switch-bool -DLEGACY_HIPBLAS_DIRECT -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM -DENABLE_CK=1 -Wno-undefined-func-template -D__HIP_PLATFORM_HCC__=1 -I/job/aiter-cache-j-3fcb0ad8a0a4/prebuild/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)