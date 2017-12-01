# C/C++ compilation involves two stages:
# - compiling: convert each source (*.cpp) file indepdendently into binary object (.o) files
# - linking: assemble all the object files into a single executable
#
# Since any C++ project contains multiple source files, the above
# process involved firing multiple commands. Makefiles help us
# write all the compilation stages in a single file and run them
# as one big script.
#
# See http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
# and many many other tutorials for makefiles


# some variables
BUILD_DIR  :=_build
EXECUTABLE := blurMask
OUTPUT := Output

# contain a list of test (blank for handout)
TESTS = $(patsubst %.cpp, %, $(wildcard test*.cpp))

# list of headers
HEADERS = $(wildcard *.h)

# the C++ compiler/linker to be used. define here so that we can change
# it easily if needed
CXX := g++ -Wall -g3 -ggdb -std=c++11 -I. -O3

# ------------------------------------------------------------------------------

# 'make' or 'make all' runs the default target 'all' which requires that
# the EXECUTABLE must be available; if the EXECUTABLE is already available
# then nothing happens. see the rules for EXECUTABLE

all: $(EXECUTABLE)

# ------------------------------------------------------------------------------

# 'make run' runs the target 'run' which calls for running the executable
# if the executable is not available, then the rules for creating it are run
# this is a superset of the target 'all' because it creates the executable
# and also runs it

run: $(EXECUTABLE)
	./$(EXECUTABLE)

# ------------------------------------------------------------------------------

# 'make clean' runs the target 'clean' which in turn removes the
# intermediate .o files and the executable

clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE) $(OUTPUT) $(TESTS)

# ------------------------------------------------------------------------------

# make tests' compiles the test cases
tests: $(TESTS)

# ------------------------------------------------------------------------------

# rule for creating the executable: this "links" the .o files using the g++ linker.
# If .o files are not available, then the rules for creating .o files are run.

$(EXECUTABLE): $(BUILD_DIR)/blurMask.o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o
	$(CXX) $(BUILD_DIR)/blurMask.o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o -o $(EXECUTABLE)
	mkdir -p $(OUTPUT)

# ------------------------------------------------------------------------------

# rules for creating the .o files:  compile each of the .cpp files and create a
# corresponding .o file. Each .o depends upon the corresponding .cpp and other .h files

$(BUILD_DIR)/%.o: %.cpp $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) -c $< -o $@
	
# ------------------------------------------------------------------------------

# rule for creating the tests: this "links" the .o files using the g++ linker.
# It crates each test .o file and links the created to all the other ones.

$(TESTS) : $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o $(HEADERS)
	$(CXX) -c $(@).cpp -o $(BUILD_DIR)/$@.o
	$(CXX) $(BUILD_DIR)/$(@).o $(BUILD_DIR)/basicImageManipulation.o $(BUILD_DIR)/filtering.o $(BUILD_DIR)/Image.o $(BUILD_DIR)/lodepng.o -o $(EXECUTABLE) -o $(@)
	mkdir -p $(OUTPUT)
