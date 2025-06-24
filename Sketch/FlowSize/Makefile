CC = g++
OPTION = -std=c++17 -g
HEADFILE = CMSketch.h CUSketch.h

main : main.cpp $(HEADFILE)
	$(CC) -o main main.cpp $(OPTION)

main_CU : main_CU.cpp $(HEADFILE)
	$(CC) -o main_CU main_CU.cpp $(OPTION)

clean : 
	rm main main_CU
