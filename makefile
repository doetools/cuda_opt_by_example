# use -Xptxas -O0 to disable any optimization by NVCC
# use -Xptxas -dlcm=ca to cache global loads into l1 cache


ALL:
	nvcc memory_coalesce.cu -o memory_coalesce  -Xptxas -O3
	./memory_coalesce
	rm memory_coalesce

UTIL:
	g++ utility.cpp -o utility
	./utility
	rm utility