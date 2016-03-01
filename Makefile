CC=gcc
CFLAGS=-fopenmp -Ofast -march=native -std=c99
BIN=matmul

SRC=complex-matmul-harness.c

all: $(SRC)
	$(CC) $(CFLAGS) -fprofile-generate -o $(BIN) $^
	./matmul 50 50 50 50
	$(CC) $(CFLAGS) -fprofile-use -fprofile-correction -o $(BIN) $^

clean:
	$(RM) $(BIN)
