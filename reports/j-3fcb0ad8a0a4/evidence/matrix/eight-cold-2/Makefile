
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -D__HIP_PLATFORM_HCC__=1 -mllvm -amdgpu-function-calls=false -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM -fno-offload-uniform-block -std=c++20 -fgpu-flush-denormals-to-zero -DUSE_PROF_API=1 -O3 -Wno-vla-cxx-extension --offload-arch=gfx942 -DENABLE_FP8 -DLEGACY_HIPBLAS_DIRECT -DENABLE_CK=1 -mllvm -amdgpu-early-inline-all=true -Wno-unused-result -U__HIP_NO_HALF_OPERATORS__ -Wno-undefined-func-template -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=0 -U__HIP_NO_HALF_CONVERSIONS__ -mllvm --amdgpu-kernarg-preload-count=16 -Wno-switch-bool -I/job/aiter-cache-j-3fcb0ad8a0a4/eight-cold-2/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)