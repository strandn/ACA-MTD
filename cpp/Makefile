include /home/nstrand/ITensor/this_dir.mk
include /home/nstrand/ITensor/options.mk

#Define Flags ----------

TENSOR_HEADERS=$(PREFIX)/itensor/all.h
CCFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS)
CCGFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS)
LIBFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBFLAGS) #-lgsl -lgslcblas
LIBGFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBGFLAGS) #-lgsl -lgslcblas

#Rules ------------------

%.o: %.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: test

debug: test-g

all: test

test: tt_sketch.o test.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) tt_sketch.o test.o -o test $(LIBFLAGS)

test-g: mkdebugdir .debug_objs/tt_sketch.o .debug_objs/test.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/tt_sketch.o .debug_objs/test.o -o test-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

clean:
	@rm -fr *.o .debug_objs test test-g
