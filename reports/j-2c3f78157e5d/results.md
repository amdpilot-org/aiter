# Raw results

## Pinned revision (AITER 3c5f0ba9 / CK 10cb6916), unpatched
RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=20/20   # BUG: one hash per launch

## Pinned revision + stabilizer 408a5bbe
RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=1/1000  # FIXED
DET M=8192 N=512 K=384:  unique_hashes=1/200
DET M=8192 N=512 K=640:  unique_hashes=1/200
DET M=8192 N=512 K=3968: unique_hashes=1/200
DET M=8192 N=512 K=4096: unique_hashes=1/1000
DET M=8192 N=512 K=4224: unique_hashes=1/200
CORR M=8192 N=512 K=4096: ck vs torch_ref err=0 (checkAllclose passed)
Correctness sweep (8 shapes): all err=0, passed

## main (2f9ee5e / CK 15e12dd) AS-IS, no modifications
RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=1/1000  # PASSING CRITERION MET
DET M=8192 N=512 K=384:  unique_hashes=1/200
DET M=8192 N=512 K=640:  unique_hashes=1/200
DET M=8192 N=512 K=3968: unique_hashes=1/200
DET M=8192 N=512 K=4096: unique_hashes=1/1000
DET M=8192 N=512 K=4224: unique_hashes=1/200
CORR M=8192 N=512 K=4096: ck vs torch_ref err=0
Correctness sweep (8 shapes): all err=0, passed
TUNED-DET M=4096 N=1536 K=7168: unique_hashes=1/100  (tuned v1 instance)
TUNED-DET M=128  N=4096 K=1280: unique_hashes=1/100  (tuned v1 instance)

## Equivalence
SHA-256(10cb6916+408a5bbe output) == SHA-256(main 15e12dd output):
4cee14cfe82d325ba739f357780c17484f474fc66610b4673e17526dbbaaf2e0

## Environment
GPU: AMD Instinct MI355X (gfx950), 256 CUs, HIP_VISIBLE_DEVICES=0
ROCm 7.2.0 (HIP 7.2.26015), torch 2.9.1+rocm7.2, triton 3.5.1+rocm7.2
Build: JIT (fresh AITER_JIT_DIR), heuristic v1 instance only (empty tune file)
       unless noted (tuned check used a 2-row tune file)
