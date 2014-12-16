/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file txt_write3.cpp Writes an integer image, a float image, a hexadecimal
 * image and a unsigned int image and tests them
 *
 *****************************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int hextest(size_t rows, size_t cols, string filename)
{
	vector<vector<unsigned int>> data(rows);
	ofstream ofs(filename.c_str());
	for(size_t rr=0; rr<rows; rr++) {
		data[rr].resize(cols);
		for(size_t cc=0; cc<cols; cc++) {
			data[rr][cc] = rand();

			// Test poorly separated data (tab+space+comma)
			if(cc != 0)
				ofs << ",   \t";
			ofs << "0x" << hex << data[rr][cc];
		}
		ofs << endl;
	}
	ofs.close();

	{
		auto reread = readMRImage(filename);
		int64_t index[2];
		for(NDIter<uint32_t> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			unsigned int v = data[index[1]][index[0]];
			if(v != *sit) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != UINT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	{
		auto reread = readNDArray(filename);
		int64_t index[2];
		for(NDIter<uint32_t> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			unsigned int v = data[index[1]][index[0]];
			if(v != *sit) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != UINT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	return 0;
}

int uinttest(size_t rows, size_t cols, string filename)
{
	ofstream ofs(filename.c_str());
	vector<vector<unsigned int>> data(rows);
	for(size_t rr=0; rr<rows; rr++) {
		data[rr].resize(cols);
		for(size_t cc=0; cc<cols; cc++) {
			data[rr][cc] = rand();

			// Test poorly separated data (space+comma)
			if(cc != 0)
				ofs << ",  ";
			ofs << data[rr][cc];
		}
		ofs << endl;
	}
	ofs.close();

	{
		auto reread = readMRImage(filename);
		int64_t index[2];
		for(NDIter<uint32_t> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			unsigned int v = data[index[1]][index[0]];
			if(v != *sit) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != UINT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	{
		auto reread = readNDArray(filename);
		int64_t index[2];
		for(NDIter<uint32_t> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			unsigned int v = data[index[1]][index[0]];
			if(v != *sit) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != UINT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	return 0;
}

int inttest(size_t rows, size_t cols, string filename)
{
	ofstream ofs(filename.c_str());
	vector<vector<int>> data(rows);
	for(size_t rr=0; rr<rows; rr++) {
		data[rr].resize(cols);
		for(size_t cc=0; cc<cols; cc++) {
			data[rr][cc] = rand()-(int)RAND_MAX/2;

			// Test comma separated data (comma)
			if(cc != 0)
				ofs << ",";
			ofs << data[rr][cc];
		}
		ofs << endl;
	}
	ofs.close();

	{
		auto reread = readMRImage(filename);
		int64_t index[2];
		for(NDIter<int> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			int v = data[index[1]][index[0]];
			if(v != *sit) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != INT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	{
		auto reread = readNDArray(filename);
		int64_t index[2];
		for(NDIter<int> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			int v = data[index[1]][index[0]];
			if(v != *sit) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != INT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	return 0;
}

int floattest(size_t rows, size_t cols, string filename)
{
	ofstream ofs(filename.c_str());
	vector<vector<double>> data(rows);
	for(size_t rr=0; rr<rows; rr++) {
		data[rr].resize(cols);
		for(size_t cc=0; cc<cols; cc++) {
			data[rr][cc] = rand()/(double)RAND_MAX;

			// Test semicolon separated
			if(cc != 0)
				ofs << ";";
			ofs << data[rr][cc];
		}
		ofs << endl;
	}
	ofs.close();

	{
		auto reread = readMRImage(filename);
		int64_t index[2];
		for(NDIter<double> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			double v = data[index[1]][index[0]];
			if(fabs(v - *sit) > 0.0001) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != FLOAT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	{
		auto reread = readNDArray(filename);
		int64_t index[2];
		for(NDIter<double> sit(reread); !sit.eof(); ++sit) {
			sit.index(2, index);
			double v = data[index[1]][index[0]];
			if(fabs(v - *sit) > 0.0001) {
				cerr << "Value Mismatch! " << setprecision(20) << *sit
					<< " vs " << setprecision(20) << v << endl;
				return -1;
			}
		}

		if(reread->type() != FLOAT32) {
			cerr << "Type Mismatch!" << endl;
			return -1;
		}
	}

	return 0;
}

int main()
{
	if(hextest(13, 3, "txt_write3_hextest.csv") != 0) {
		cerr << "Hextest Failed!" << endl;
		return -1;
	}

	if(uinttest(11, 2, "txt_write3_uinttest.csv") != 0) {
		cerr << "Uint Test Failed!" << endl;
		return -1;
	}

	if(inttest(1, 23, "txt_write3_inttest.csv") != 0) {
		cerr << "Int Test Failed!" << endl;
		return -1;
	}

	if(floattest(12, 1, "txt_write3_floattest.csv") != 0) {
		cerr << "Float Test Failed!" << endl;
		return -1;
	}

}

