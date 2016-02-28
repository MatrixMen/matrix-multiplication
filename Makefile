CC=gcc
CFLAGS=-fopenmp -Ofast -march=native
BIN=matmul

SRC=complex-matmul-harness.c

all: $(SRC)
	$(CC) $(CFLAGS) -o $(BIN) $^

clean:
	$(RM) $(BIN)
