CC=gcc
CFLAGS=-Wall -Wextra -Wuninitialized -MMD -g -fdiagnostics-color=auto
LDFLAGS=
SRC=$(wildcard src/*.c)
OBJ=$(subst src/,build/,$(SRC:.c=.o))
DEP=$(subst src/,build/,$(SRC:.c=.d))

prog: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ -c $<

doc: doc/nn.tex
	mkdir -p doc/out
	cd doc; pdflatex -output-directory out nn.tex

clean:
	rm -rf build

-include $(DEP)
