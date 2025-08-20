# Makefile for C++ tests
CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I../../src/csrc -I/opt/homebrew/include
LIBS = -L/opt/homebrew/lib -lgtest -lgtest_main -lpthread

SOURCES = test_ma_core.cpp ../../src/csrc/ma_core.cpp
TARGET = ma_core_tests

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LIBS) -o $(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# Alternative build using CMake
cmake-build:
	mkdir -p build
	cd build && cmake .. && make

cmake-test: cmake-build
	cd build && ./ma_core_tests

cmake-clean:
	rm -rf build