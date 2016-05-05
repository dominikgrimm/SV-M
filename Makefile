clean:
	rm -rf build
	rm -rf bin

createdirs:
	mkdir -p build
	mkdir -p bin

all: sv-m

globals.o: src/globals.cc
	g++ -Wall -c -o build/globals.o src/globals.cc

configs.o: src/configs.cc
	g++ -Wall -c -o build/configs.o src/configs.cc

sv-m.o: src/sv-m.cc
	g++ -Wall -c -o build/sv-m.o src/sv-m.cc

svm.o: src/svm.cpp
	g++ -Wall -c -o build/svm.o src/svm.cpp

svmclass.o: src/svmclass.cc
	g++ -Wall -c -o build/svmclass.o src/svmclass.cc

StringHelper.o: src/StringHelper.cc
	g++ -Wall -c -o build/StringHelper.o src/StringHelper.cc

svm_predict.o: src/svm_predict.cc
	g++ -Wall -c -o build/svm_predict.o src/svm_predict.cc

sv-m:  createdirs sv-m.o StringHelper.o svm.o svmclass.o globals.o configs.o
	g++ -o bin/sv-m build/sv-m.o build/StringHelper.o build/svm.o build/svmclass.o build/globals.o build/configs.o
