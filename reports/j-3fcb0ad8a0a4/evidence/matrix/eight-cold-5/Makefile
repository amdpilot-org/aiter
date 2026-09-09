
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -mllvm -amdgpu-function-calls=false -mllvm --lsr-drop-solution=1 -O3 -DENABLE_FP8 -D__HIP_PLATFORM_HCC__=1 -Wno-switch-bool -fno-offload-uniform-block -U__HIP_NO_HALF_CONVERSIONS__ -mllvm --amdgpu-kernarg-preload-count=16 -mllvm -amdgpu-early-inline-all=true -U__HIP_NO_HALF_OPERATORS__ -DUSE_PROF_API=1 -fgpu-flush-denormals-to-zero --offload-arch=gfx942 -std=c++20 -DLEGACY_HIPBLAS_DIRECT -Wno-unused-result -DENABLE_CK=1 -D__HIP_PLATFORM_AMD__=1 -Wno-vla-cxx-extension -Wno-undefined-func-template -mllvm -enable-post-misched=0 -DUSE_ROCM -I/job/aiter-cache-j-3fcb0ad8a0a4/eight-cold-5/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)