# Top level make file
# ----------------------------------------------------------------------------
# Authors:
#  James Knight
# ----------------------------------------------------------------------------
# Code derived from Andrew Mundy and Andrew Rowley, University of Manchester

APPS = conv_layer

all :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" ) || exit $$?; done
	for a in $(APPS); do ( cd $$a; "$(MAKE)" PROFILER_ENABLED=1 ) || exit $$?; done

clean :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean ) || exit $$?; done
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean PROFILER_ENABLED=1 ) || exit $$?; done
