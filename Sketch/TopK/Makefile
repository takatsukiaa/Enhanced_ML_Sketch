CC = g++
OPTION = -std=c++17

OBJS = main.o TopK.o CUSketch.o CMSketch.o common.o

main: $(OBJS)
	$(CC) -o main $(OBJS) $(OPTION)

main.o: main.cpp TopK.h CUSketch.h CMSketch.h common.h
	$(CC) -c main.cpp $(OPTION)

TopK.o: TopK.cpp TopK.h CUSketch.h CMSketch.h
	$(CC) -c TopK.cpp $(OPTION)

CUSketch.o: CUSketch.cpp CUSketch.h common.h
	$(CC) -c CUSketch.cpp $(OPTION)

CMSketch.o: CMSketch.cpp CMSketch.h common.h
	$(CC) -c CMSketch.cpp $(OPTION)

common.o: common.cpp common.h
	$(CC) -c common.cpp $(OPTION)

clean:
	rm -f *.o main