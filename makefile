# Compiler and flags
CC = gcc
CFLAGS = -g -Wall -pg -Wextra -Werror -I. 
LDFLAGS = -lm

# Source files
SRCS = main.c 
OBJS = $(SRCS:.c=.o)

# Targets
all: build/test


build/test: $(OBJS)
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f  $(OBJS) build/test
	rmdir build

.PHONY: all clean
