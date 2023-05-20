.PHONY: default
default: all

SRC_FILE    = main.cpp
OUT_NAME    = $(SRC_FILE).o
INCLUDE_DIR = -I/opt/homebrew/Cellar/opencv/4.7.0_4/include/opencv4
override CFLAGS    += -O2 -v -std=c++11
LIBRARY_DIR = -L/opt/homebrew/Cellar/opencv/4.7.0_4/lib
LIBRARIES   = -lopencv_core -lopencv_imgproc -lopencv_highgui    \
              -lopencv_video -lopencv_videoio -lopencv_imgcodecs \
		
COMPILER    = clang++	  

all: 
	$(COMPILER) $(INCLUDE_DIR) $(CFLAGS) $(SRC_FILE) -o $(OUT_NAME) $(LIBRARY_DIR) $(LIBRARIES)
 
clean:
	$(RM) $(OUT_NAME)
