#-------------------------------------------------------------------------------
# Sample makefile for building the code samples. Read inline comments for
# documentation.
#
# Original from 
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------

# The following variables will likely need to be customized, depending on where
# and how you built LLVM & Clang. They can be overridden by setting them on the
# make command line: "make VARNAME=VALUE", etc.

# LLVM_SRC_PATH is the path to the root of the checked out source code. This
# directory should contain the configure script, the include/ and lib/
# directories of LLVM, Clang in tools/clang/, etc.
#

LLVM_SRC_PATH := /usr/lib/llvm-8

# LLVM_BUILD_PATH is the directory in which you built LLVM - where you ran
# configure or cmake.

LLVM_BUILD_PATH := $(LLVM_SRC_PATH)

LLVM_BIN_PATH 	:= $(LLVM_BUILD_PATH)/bin

$(info -----------------------------------------------)
$(info Using LLVM_SRC_PATH = $(LLVM_SRC_PATH))
$(info Using LLVM_BUILD_PATH = $(LLVM_BUILD_PATH))
$(info Using LLVM_BIN_PATH = $(LLVM_BIN_PATH))
$(info -----------------------------------------------)

# CXX has to be a fairly modern C++ compiler that supports C++11. gcc 4.8 and
# higher or Clang 3.2 and higher are recommended. Best of all, if you build LLVM
# from sources, use the same compiler you built LLVM with.
# Note: starting with release 3.7, llvm-config will inject flags that gcc may
# not support (for example '-Wcovered-switch-default'). If you run into this
# problem, build with CXX set to a modern clang++ binary instead of g++.

#CXX := g++-5
CXX := $(LLVM_BIN_PATH)/clang++
CXXFLAGS := -g  -fno-rtti

LLVM_CXXFLAGS := `$(LLVM_BIN_PATH)/llvm-config --cxxflags`
LLVM_LDFLAGS := `$(LLVM_BIN_PATH)/llvm-config --ldflags --libs --system-libs`

# These are required when compiling vs. a source distribution of Clang. For
# binary distributions llvm-config --cxxflags gives the right path.

CLANG_INCLUDES := \
	-I$(LLVM_SRC_PATH)/tools/clang/include \
	-I$(LLVM_BUILD_PATH)/tools/clang/include

# List of Clang libraries to link. The proper -L will be provided by the
# call to llvm-config
# Note that I'm using -Wl,--{start|end}-group around the Clang libs; this is
# because there are circular dependencies that make the correct order difficult
# to specify and maintain. The linker group options make the linking somewhat
# slower, but IMHO they're still perfectly fine for tools that link with Clang.
CLANG_LIBS := \
	-Wl,--start-group \
	-lclangAST \
	-lclangAnalysis \
	-lclangBasic \
	-lclangDriver \
	-lclangEdit \
	-lclangFrontend \
	-lclangFrontendTool \
	-lclangLex \
	-lclangParse \
	-lclangSema \
	-lclangEdit \
	-lclangASTMatchers \
	-lclangRewrite \
	-lclangRewriteFrontend \
	-lclangStaticAnalyzerFrontend \
	-lclangStaticAnalyzerCheckers \
	-lclangStaticAnalyzerCore \
	-lclangSerialization \
	-lclangToolingCore \
	-lclangTooling \
	-Wl,--end-group

# Internal paths in this project: where to find sources, and where to put
# build artifacts.

SRCDIR := src
BUILDDIR := build

HEADERS = \
  $(SRCDIR)/transformer.h \
  $(SRCDIR)/myastvisitor.h \
  $(SRCDIR)/srcbuf.h \
  $(SRCDIR)/stringops.h \
  $(SRCDIR)/optionsparser.h \
  $(SRCDIR)/specialization_db.h


OBJECTS = \
  $(BUILDDIR)/transformer.o \
  $(BUILDDIR)/stringops.o \
  $(BUILDDIR)/codegen.o \
  $(BUILDDIR)/srcbuf.o \
  $(BUILDDIR)/myastvisitor.o \
  $(BUILDDIR)/loop_function.o \
  $(BUILDDIR)/optionsparser.o \
  $(BUILDDIR)/specialization_db.o \
  $(BUILDDIR)/write_output.o


$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp ${HEADERS} Makefile
	$(CXX) -c $(CXXFLAGS) $(LLVM_CXXFLAGS) $(CLANG_INCLUDES) $< -o $@

$(BUILDDIR)/transformer: ${OBJECTS}
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $(OBJECTS) \
		$(CLANG_LIBS) $(LLVM_LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/* *.dot test/*.pyc test/__pycache__
