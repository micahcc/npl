#include <iostream>
#include <iomanip>
#include <ctime>

#include <linalg.h>

using std::cout;
using std::setw;
using std::endl;

const int VERBOSE = false;

int main(int argc, char** argv)
{

	int size = 0;
	if(argc != 2) {
		cout << "Please provide a size of input" << endl;
		return -1;
	} else {
		size = atoi(argv[1]);
		cout << "Array Size: " << size << endl;
	}

	alglib::real_2d_array mat;
	alglib::hqrndstate state;
	alglib::hqrndrandomize(state);
	mat.setlength(size, size);
	for(int rr = 0 ; rr < mat.rows(); rr++) {
		for(int cc = 0 ; cc < mat.cols(); cc++) {
			mat[rr][cc] = mat[cc][rr] = alglib::hqrndnormal(state);
		}
	}
	
	if(VERBOSE) {
		cout << "Matrix: " << endl;
		for(int rr = 0 ; rr < mat.rows(); rr++) {
			for(int cc = 0 ; cc < mat.cols(); cc++) {
				cout << setw(10) << mat[rr][cc];
			}
			cout << endl;
		}
		cout << endl;
	}

	alglib::real_1d_array d;
	alglib::real_2d_array z;
	auto t = clock();
	alglib::smatrixevd(mat, mat.rows(), 1, 0, d, z);
	t = clock() - t;
	
	cout << (double)t/CLOCKS_PER_SEC << "s" << endl;
	
	if(VERBOSE) {
		for(int cc = 0 ; cc < mat.cols(); cc++) {
			cout << "lambda: " << d[cc] << endl;
			cout << "V: ";
			for(int rr = 0 ; rr < mat.rows(); rr++) {
				cout << setw(10) << z[rr][cc];
			}
			cout << endl;
		}
	}
}
