all: test_gcc test_clang

test_gcc: test.cpp
	g++ -march=native -O3 test.cpp -o test_gcc
	objdump -d test_gcc > test_gcc.txt

test_clang: test.cpp
	clang++ -march=native -O3 test.cpp -o test_clang
	objdump -d test_clang > test_clang.txt

clean:
	rm -f test_gcc test_gcc.txt test_clang test_clang.txt
