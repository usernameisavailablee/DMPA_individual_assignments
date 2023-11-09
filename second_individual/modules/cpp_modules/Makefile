CC = g++
CFLAGS = -std=c++11
LIBS = `pkg-config --libs opencv4`
INCLUDES = `pkg-config --cflags opencv4`

all: main

main: main.cpp
	$(CC) $(CFLAGS) -o main main.cpp $(LIBS) $(INCLUDES)

clean:
	rm -f main
