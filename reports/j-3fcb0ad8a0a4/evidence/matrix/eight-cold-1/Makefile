
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -U__HIP_NO_HALF_CONVERSIONS__ -DUSE_PROF_API=1 -DENABLE_FP8 -D__HIP_PLATFORM_HCC__=1 -Wno-undefined-func-template -Wno-vla-cxx-extension -std=c++20 -mllvm -amdgpu-early-inline-all=true -DUSE_ROCM -fgpu-flush-denormals-to-zero -mllvm --lsr-drop-solution=1 -mllvm --amdgpu-kernarg-preload-count=16 -mllvm -enable-post-misched=0 -U__HIP_NO_HALF_OPERATORS__ -Wno-switch-bool -fno-offload-uniform-block -O3 -DLEGACY_HIPBLAS_DIRECT -mllvm -amdgpu-function-calls=false -DENABLE_CK=1 -D__HIP_PLATFORM_AMD__=1 -Wno-unused-result --offload-arch=gfx942 -I/job/aiter-cache-j-3fcb0ad8a0a4/eight-cold-1/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)