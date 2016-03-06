# compiler
CC = mpicc

# source
SRC = pgmio.c pgm_reconstructor_serial.c

# object
OBJ = $(SRC:.c=.o)

# output executable file name
OUTPUT = pgm_reconstructor_serial

$(OUTPUT) : $(OBJ)
	$(CC) -o $@ $^

%.o : %.c
	$(CC) -c $^

.PHONY : clean
clean :
	rm $(OUTPUT) $(OBJ)
