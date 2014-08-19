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
 * @file utility.cpp
 *
 *****************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include "utility.h"

#include <string>
#include <cassert>
#include <list>
#include <vector>

using std::endl;
using std::string;
using std::list;
using std::vector;
using std::cerr;
using std::numeric_limits;

namespace npl {

/**
 * @brief Reads a file and returns true if its entirely made up of printable ascii
 *
 * @param filename name of file to read
 *
 * @return if the file is text
 */
bool isTxt(string filename)
{
	std::ifstream ifs(filename.c_str());
	if(!ifs.is_open()) {
		return false;
	}

	char a = ifs.get();
	while(!ifs.eof()) {
		if(a != '\t' && a != '\n' && a != '\r' && !(a <= 126 && a >= 32)) {
			return false;
		}
		a = ifs.get();
	}

	return true;
}

/**
 * @brief Returns true if a file exists and is possible to open
 *
 * @param filename to read
 *
 * @return if the file exists
 */
bool fileExists(std::string filename)
{
	std::ifstream f(filename.c_str());
	if (f.good()) {
		f.close();
		return true;
	} else {
		f.close();
		return false;
	}
}

/**
 * @brief Removes whitespace at the beginning and end of a string
 *
 * @param str string to chomp
 *
 * @return chomped string
 */
string chomp(string str)
{
	int begin, end;
	for(begin = 0; begin < (int)str.size() && isspace(str[begin]); begin++)
		continue;

	if(begin == (int)str.size())
		return string("");
	
	for(end = str.size()-1; end >= 0 && isspace(str[end]); end--)
		continue;

	if(end < 0) //this can't happen
		end = str.length()-1;
	
	return str.substr(begin, end-begin+1);
}

/**
 * @brief Given a delimiter splits the line based on the delmiter and removes
 * extra white space as necessary
 *
 * Note that repeated white space characters will be removed but other delimiters
 * wont be
 *
 * TODO handle values inside quoates like ", " which should technically be
 * ignored and maybe \, but not \\, etc
 *
 * @param line string holding the line
 * @param delim character delimiter[s]
 *
 * @return vector of strings, one string per parsed token
 */
vector<string> parseLine(std::string line, string delim)
{
	size_t pos = 0;
	size_t prev = 0;
	vector<string> out;
	while(pos < line.length()) {
		prev = pos;
		pos = line.find_first_of(delim, prev);
		if(pos == string::npos) {
			pos = line.length();
			string tmp = chomp(line.substr(prev));
			if(tmp.length() != 0)
				out.push_back(tmp);
		} else {
			string tmp = chomp(line.substr(prev, pos-prev));
			if(tmp.length() != 0)
				out.push_back(tmp);
			
			//move past delimiter
			pos++;

			//skip remaining white space
			while(pos < line.size() && isspace(line[pos]))
				pos++;
		}
	}
	return out;
};

/**
 * @brief This function parses an input and returns a list of rows
 *
 * This function reads a text file of the form:
 * DvDvDvD....Dv[D]
 * by deciding whether white space or commas or semi-colons is
 * the delimiter and then proceeding to read each line. It does
 * this by looking at the first 10 lines and comparing the line
 * based on each possible deliminter.
 * out
 *
 * @param filename file to read
 * @param comment lines with this first non-white space character will be ignored
 *
 * @return Vector of vectors, where outer vectors are rows, inner are columns
 */
vector<vector<string>> readStrCSV(string filename, char comment)
{
    std::ifstream fin(filename.c_str());

	if(!fin.is_open()) {
		cerr << "Couldn't open:  " << filename << endl;
		return vector<vector<string>>();
	}

    std::string line;
	vector<string> tmparr;

	list<vector<string> > outstore;

	int linenum = 0;
	int minwidth = numeric_limits<int>::max();
	int maxwidth = 0;
	int priority = 2;

	string delims[3] = {";", "\t ", ","};

	/* Start trying delimiters. Priority is in reverse order so the last that
	 * grants the same number of outputs on a line and isn't 1 is given the
	 * highest priority
	 */
	
	//grab the first few lines
	list<string> firstlines;
	for(int ii = 0 ; !fin.eof(); ii++ ) {
		getline(fin, line);
		firstlines.push_back(line);
	}

	//test our possible delimiters
	for(int ii = 0 ; ii < 3 ; ii++) {
		list<string>::iterator it = firstlines.begin();
		minwidth = numeric_limits<int>::max();
		maxwidth = 0;
		for(;it != firstlines.end(); it++) {
			line = *it;
			string tmp = chomp(line);
			if(line[0] == comment || tmp[0] == comment || tmp.size() == 0)
				continue;

			// parse the line, and compute width
			tmparr = parseLine(line, delims[ii]);

			if((int)tmparr.size() < minwidth) {
				minwidth = tmparr.size();
			}
			if((int)tmparr.size() > maxwidth) {
				maxwidth = tmparr.size();
			}
		}
		if(maxwidth > 1 && maxwidth == minwidth) {
			priority = ii;
		}
	}
			
	
	//re-process first 10 lines using the proper delimiter
	list<string>::iterator it = firstlines.begin();
	minwidth = numeric_limits<int>::max();
	maxwidth = 0;
	for(;it != firstlines.end(); it++, linenum++) {
		line = *it;
		string tmp = chomp(line);
		if(line[0] == comment || tmp[0] == comment || tmp.size() == 0) {
			continue;
		}

		tmparr = parseLine(line, delims[priority]);
		if((int)tmparr.size() < minwidth)
			minwidth = tmparr.size();
		if((int)tmparr.size() > maxwidth)
			maxwidth = tmparr.size();

		outstore.push_back(tmparr);
	}

	//process the rest of the input (the reason we don't use get is we might want
	//to parse - as stdin
	for(;!fin.eof(); linenum++) {
		getline(fin, line);
		string tmp = chomp(line);
		if(line[0] == comment || tmp[0] == comment || tmp.size() == 0) {
			continue;
		}
		
		tmparr = parseLine(line, delims[priority]);
		if((int)tmparr.size() < minwidth) {
			minwidth = tmparr.size();
		}
		if((int)tmparr.size() > maxwidth) {
			maxwidth = tmparr.size();
		}

		outstore.push_back(tmparr);
	}

	//copy the output from a list to a vector
	vector<vector<string>> out(outstore.size());

	size_t ii=0;
	for(auto it = outstore.begin(); it != outstore.end(); ++ii, ++it) {
		out[ii] = std::move(*it);
	}

	if(minwidth != maxwidth || minwidth == 0) {
		cerr << "Warning you may want to be concerned that there are "
			<< "differences in the number of fields per line" << endl;
	}

    return out;
}

/**
 * @brief This function parses an input and returns a list of rows
 *
 * This function reads a text file of the form:
 * DvDvDvD....Dv[D]
 * by deciding whether white space or commas or semi-colons is
 * the delimiter and then proceeding to read each line. It does
 * this by looking at the first 10 lines and comparing the line
 * based on each possible deliminter.
 * out
 *
 * @param filename file to read
 * @param out vector of rows (stored in vectors)
 * @param comment lines with this first non-white space character will be ignored
 *
 * @return maximum row width
 */
std::vector<std::vector<double>> readNumericCSV(string filename, char comment)
{

	vector<vector<string>> tmp = readStrCSV(filename, comment);
	
	//allocate output data
	std::vector<std::vector<double>> out(tmp.size());
	for(unsigned int rr = 0 ; rr < out.size(); rr++) {
		out[rr].resize(tmp[rr].size(), 0);
		for(int cc = 0 ; cc < tmp[rr].size(); cc++) {
			out[rr][cc] = atof(tmp[rr][cc].c_str());
		}
	}
	
	return out;
}

}
