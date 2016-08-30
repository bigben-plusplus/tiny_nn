# Third-party libraries
ARMAROOT    ?= C:\Program Files\armadillo
BLASROOT    ?= D:\Bigben\usr\lib\Blas\i686

BLASLIBS    ?= -lopenblas

CC           = g++
CCFLAGS      = -I.
ifdef Debug
CCFLAGS     += -g -ggdb
else
CCFLAGS     += -O2
endif

CCFLAGS     += -I"$(ARMAROOT)/include" -DARMA_USE_LAPACK -DARMA_USE_BLAS
CCFLAGS     += -L"$(BLASROOT)"

DEMO_SRC = example/iris_classify.cpp
DEMO_OBJ = $(patsubst %.cpp,%.o,$(DEMO_SRC))

all: test

$(DEMO_OBJ): $(DEMO_SRC)
	$(CC) $(CCFLAGS) -Isrc -o $@ -c $<

example: $(DEMO_OBJ)	
	$(CC) $(CCFLAGS) -o $(patsubst %.o,%.exe,$(^F)) $(DEMO_OBJ) -lopenblas

format:
	@astyle style=kr indent=spaces=2 -p -U --recursive --suffix=none "*.hpp" "*.h" "*.cpp" "*.c"

test: clean example
	@iris_classify.exe -k 450 -r 1.2
	
clean:
	@-rm -r $(DEMO_OBJ) *.exe
	@-rm *.log *.dot *.json *.txt
