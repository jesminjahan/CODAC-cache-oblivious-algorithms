CC=icpc
CFLAGS=-O3 -Wall -Werror -ansi-alias -ip -opt-subscript-in-range -opt-prefetch=4 -fomit-frame-pointer -funroll-all-loops -restrict -vec-report -parallel -xhost

all: 
	$(CC) $(CFLAGS) -o pbfswsldq BFS_WSLDQ.cpp 

clean:
	rm -f pbfswsldq
