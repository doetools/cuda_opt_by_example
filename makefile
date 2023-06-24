# use -Xptxas -O0 to disable any optimization by NVCC
# use -Xptxas -dlcm=ca to cache global loads into l1 cache
current_dir = $(shell pwd)
PROF_FILE_NAME = ./profile.md

ALL:
	nvcc memory_coalesce.cu -o memory_coalesce  -Xptxas -O3
	./memory_coalesce
	rm memory_coalesce

PROFILE:
	nvcc memory_coalesce.cu -o memory_coalesce  -Xptxas -O3
	nvprof --log-file $(PROF_FILE_NAME) ./memory_coalesce
	rm memory_coalesce

UTIL:
	g++ utility.cpp -o utility
	./utility
	rm utility