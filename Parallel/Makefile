# compiler
CC = mpicc

# source
SRC = pgmio.c pgm_reconstructor_parallel.c

# object
OBJ = $(SRC:.c=.o)

# output executable file name
OUTPUT = pgm_reconstructor_parallel

$(OUTPUT) : $(OBJ)
	$(CC) -fastsse -o $@ $^

%.o : %.c
	$(CC) -fastsse -c $^

.PHONY : clean
clean :
	rm $(OUTPUT) $(OBJ)
