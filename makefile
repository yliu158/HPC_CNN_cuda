run: serial.o fclforward.o link
	./test

serial.o: serial.cpp main.h
	g++ -std=c++17 -c serial.cpp main.h

fclforward.o: fclforward.cu main.h
	nvcc -c fclforward.cu

link:
	nvcc fclforward.o serial.o -o test

clean:
	rm -fr *.o test
