
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -mllvm -amdgpu-early-inline-all=true -mllvm --amdgpu-kernarg-preload-count=16 -U__HIP_NO_HALF_CONVERSIONS__ -O3 -DUSE_PROF_API=1 -Wno-undefined-func-template -std=c++20 -Wno-unused-result -DENABLE_FP8 -fno-offload-uniform-block -mllvm -enable-post-misched=0 -U__HIP_NO_HALF_OPERATORS__ -mllvm --lsr-drop-solution=1 -Wno-vla-cxx-extension -DUSE_ROCM -DENABLE_CK=1 --offload-arch=gfx942 -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -DLEGACY_HIPBLAS_DIRECT -Wno-switch-bool -mllvm -amdgpu-function-calls=false -fgpu-flush-denormals-to-zero -I/job/aiter-cache-j-3fcb0ad8a0a4/eight-cold-6/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)