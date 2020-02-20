NVCC = nvcc
CFLAGS = -g -G -O0

EasyBMP.o: ./bmp/EasyBMP.cpp
	g++ -O3 -c $< -o $@

LNK = ./bmp/EasyBMP.h ./bmp/EasyBMP.cpp
salt_and_pepper: kernel.cu main.h
	$(NVCC) $(CFLAGS) $< -o $@ EasyBMP.o

clear:
	rm output.bmp