CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -DNDEBUG -Wall -Wextra -Wpedantic -Werror

all: morph_envelope

morph_envelope: src/morph_envelope.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f morph_envelope morph_envelope_ubsan morph_envelope_asan

.PHONY: all clean
