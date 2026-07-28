
# Following line(s) are printed with "make help".  Use columns 8 and 30
#% suN_gauge_surface_twist make options 
#%     NCOL=<N>              - SU(N) gauge simulation program (default: 3)
#%     SUN_OVERRELAX_dFJ=1   - use deForcrand-Jahn full overrelax (default: su2 subgroups)
#%     GFLOWS=<K>            - Gauge action for gradient flow:
#%                                                   K=0: "WILSON"
#%                                                   K=1: "Bulk-Prevention"
#%                                                   K=2: "Luscher-Weisz"
#%                                                   K=3: "IWASAKI"         (default)
#%  

# Give the location of the top level distribution directory wrt. this location.
# Can be absolute or relative
HILA_DIR := ../..

# Number of colors
NCOL := 4

# Gauge action for gradient flow update: 
# "WILSON": 0, "BP": 1, "LW": 2, "IWASAKI": 3, "DBW2": 4
GFLOWS := 3

# Set default goal and arch
.DEFAULT_GOAL := suN_gauge_surface_twist

ifndef ARCH
ARCH := vanilla
endif

# Read in the main makefile contents, incl. platforms
include $(HILA_DIR)/libraries/main.mk

#HILAPP_OPTS += -insert-includes
#HILAPP_OPTS += -comment-pragmas
HILAPP_OPTS += -check-init

APP_OPTS += -DNDIM=4 -DNCOLOR=${NCOL} -DGFLOWACTION=${GFLOWS}

ifdef SUN_OVERRELAX_dFJ
APP_OPTS += -DSUN_OVERRELAX_dFJ
endif

# Now the linking step for each target executable

SUFF=
ifdef DEBUG
ifeq (${DEBUG},1)
SUFF=_debug
endif
endif

suN_gauge_surface_twist: build/suN_gauge_surface_twist.o $(HILA_OBJECTS) $(HEADERS)
	$(LD) -o build/su${NCOL}_gauge_surface_twist${SUFF} build/suN_gauge_surface_twist.o $(wildcard *.o) $(HILA_OBJECTS) $(LDFLAGS) $(LDLIBS) $(FFTW_LIB) -lfftw3

suN_gauge_surface_twist_gf: build/suN_gauge_surface_twist_gf.o $(HILA_OBJECTS) $(HEADERS)
	$(LD) -o build/su${NCOL}_gauge_surface_twist_fs${GFLOWS}${SUFF} build/suN_gauge_surface_twist_gf.o $(wildcard *.o) $(HILA_OBJECTS) $(LDFLAGS) $(LDLIBS) $(FFTW_LIB) -lfftw3
