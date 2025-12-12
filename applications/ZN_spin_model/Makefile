
# Following line(s) are printed with "make help".  Use columns 8 and 30
#%     make [ZN_spin_model] - build ZN_spin_model program
#% ZN_spin_model make options 
#%     NCOL=<N>              - Z_N spin field (default: 3)
#%     NDIM=D                - number of dimensions D of the lattice
#% 

# Give the location of the top level distribution directory wrt. this location.
# Can be absolute or relative
HILA_DIR := ../..

# Number of colors
NCOL := 3
# Number of dimensions
NDIM := 3

ifndef ARCH
ARCH := vanilla
endif

# Set default goal and arch
.DEFAULT_GOAL := ZN_spin_model

# Read in the main makefile contents, incl. platforms
include $(HILA_DIR)/libraries/main.mk

#HILAPP_OPTS += -insert-includes
#HILAPP_OPTS += -comment-pragmas
HILAPP_OPTS += -check-init

APP_OPTS += -DNDIM=${NDIM} -DNCOLOR=${NCOL} -DGENGATHER


# With multiple targets we want to use "make target", not "make build/target".
# This is needed to carry the dependencies to build-subdir

ZN_spin_model: build/Z${NCOL}_spin_model_d${NDIM}_${ARCH} ; @:

# Now the linking step for each target executable
build/Z${NCOL}_spin_model_d${NDIM}_${ARCH}: Makefile build/ZN_spin_model.o $(HILA_OBJECTS) $(HEADERS)
	$(LD) -o $@ build/ZN_spin_model.o $(HILA_OBJECTS) $(LDFLAGS) $(LDLIBS) 








