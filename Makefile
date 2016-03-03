CC = gcc
override CFLAGS += -fopenmp -O3 -march=native -std=gnu99
BIN = matmul
PROF_BIN = profbin

SRC = team_matmul.c complex-matmul-harness.c
PROFILE_FILES = $(wildcard *.gcda)

.PHONY : all profile clean

all: clean $(SRC) profile
	$(CC) $(CFLAGS) -fprofile-use -fprofile-correction -o $(BIN) $(SRC)

clean:
	$(RM) $(BIN)
	$(RM) $(PROFILE_FILES)
	$(RM) $(PROF_BIN)

profile: profile.c team_matmul.c
	@$(CC) $(CFLAGS) -fprofile-generate -o $(PROF_BIN) $^
	@-./$(PROF_BIN) > /dev/null 2>&1

