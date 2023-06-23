# use -Xptxas -O0 to disable any optimization by NVCC

ALL:
	nvcc memory_coalesce.cu -o memory_coalesce  -Xptxas -O0
	./memory_coalesce
	rm memory_coalesce

UTIL:
	g++ utility.cpp -o utility
	./utility
	rm utility