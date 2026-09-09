
CXX=hipcc
TARGET=lib.so

SRCS = top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7.cpp
OBJS = $(SRCS:.cpp=.o)

build: $(OBJS)
	$(CXX) -shared $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -fPIC -mllvm -amdgpu-function-calls=false -mllvm --lsr-drop-solution=1 -Wno-switch-bool -mllvm -amdgpu-early-inline-all=true -DLEGACY_HIPBLAS_DIRECT -Wno-undefined-func-template -std=c++20 -O3 -fgpu-flush-denormals-to-zero -U__HIP_NO_HALF_CONVERSIONS__ -Wno-unused-result -U__HIP_NO_HALF_OPERATORS__ -DENABLE_FP8 -fno-offload-uniform-block -DUSE_ROCM -mllvm -enable-post-misched=0 -DENABLE_CK=1 -Wno-vla-cxx-extension -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -mllvm --amdgpu-kernarg-preload-count=16 --offload-arch=gfx942 -D__HIP_PLATFORM_HCC__=1 -I/job/aiter-cache-j-3fcb0ad8a0a4/eight-cold-4/build/top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7/include -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)