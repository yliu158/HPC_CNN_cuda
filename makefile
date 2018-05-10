run: serial.o device.o link

serial.o: serial.cpp main.h
	g++ -std=c++17 -c serial.cpp main.h

device.o: device.cu main.h
	nvcc -arch sm_20 -c device.cu

link:
	nvcc device.o serial.o -o test

clean:
	rm -fr *.o test
