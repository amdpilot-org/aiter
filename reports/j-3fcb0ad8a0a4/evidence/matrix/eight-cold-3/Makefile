
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -DUSE_ROCM -Wno-switch-bool -mllvm -amdgpu-function-calls=false -Wno-vla-cxx-extension -mllvm -enable-post-misched=0 -D__HIP_PLATFORM_HCC__=1 -Wno-unused-result -DENABLE_CK=1 -mllvm --lsr-drop-solution=1 -DUSE_PROF_API=1 -U__HIP_NO_HALF_OPERATORS__ -std=c++20 -mllvm -amdgpu-early-inline-all=true -D__HIP_PLATFORM_AMD__=1 -O3 --offload-arch=gfx942 -fno-offload-uniform-block -Wno-undefined-func-template -fgpu-flush-denormals-to-zero -DENABLE_FP8 -U__HIP_NO_HALF_CONVERSIONS__ -DLEGACY_HIPBLAS_DIRECT -mllvm --amdgpu-kernarg-preload-count=16 -I/job/aiter-cache-j-3fcb0ad8a0a4/eight-cold-3/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)