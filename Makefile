CC = gcc
override CFLAGS += -fopenmp -O3 -march=native -std=gnu99
BIN = matmul

SRC = complex-matmul-harness.c
PROFILE_FILES = $(wildcard *.gcda)

all: $(SRC)
	@$(CC) $(CFLAGS) -fprofile-generate -o $(BIN) $^
	@-./matmul 100 100 100 100 > /dev/null 2>&1
	$(CC) $(CFLAGS) -fprofile-use -fprofile-correction -o $(BIN) $^

clean:
	$(RM) $(BIN)
	$(RM) $(PROFILE_FILES)
