CC = g++
OPTION = -std=c++17
HEADFILE = CMSketch.h CMSACSketch.h CUSketch.h CUSACSketch.h PSketch.h PSACSketch.h

main : main.cpp $(HEADFILE)
	$(CC) -o main main.cpp $(OPTION)

main_CU : main_CU.cpp $(HEADFILE)
	$(CC) -o main_CU main_CU.cpp $(OPTION)

clean : 
	rm main

flows : Flow_Measurements.cpp $(HEADFILE)
	$(CC) -o Flow_Measurements Flow_Measurements.cpp $(OPTION)