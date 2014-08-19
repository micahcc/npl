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
 * @file utility.h
 *
 *****************************************************************************/
#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <string>
#include <cmath>
#include <list>
#include <vector>


namespace npl {

/**
 * @brief Reads a file and returns true if its entirely made up of printable ascii
 *
 * @param filename name of file to read
 *
 * @return if the file is text
 */
bool isTxt(std::string filename);

/**
 * @brief Returns true if a file exists and is possible to open
 *
 * @param filename to read
 *
 * @return if the file exists
 */
bool fileExists(std::string filename);

/**
 * @brief Removes whitespace at the beginning and end of a string
 *
 * @param str string to chomp
 *
 * @return chomped string
 */
std::string chomp(std::string str);

/**
 * @brief Given a delimiter splits the line based on the delmiter and removes
 * extra white space as necessary
 *
 * Note that repeated white space characters will be removed but other delimiters
 * wont be
 *
 * TODO handle ", " which should technically be ignored and maybe \, but not \\,
 * etc
 *
 * @param line string holding the line
 * @param delim character delimiter[s]
 *
 * @return vector of strings, one string per parsed token
 */
std::vector<std::string> parseLine(std::string line, std::string delim);

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
 * @return out vector of rows (stored in vectors)
 */
std::vector<std::vector<std::string>> readStrCSV(std::string filename,
			char comment = '#');

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
 * @return out vector of rows (stored in vectors)
 */
std::vector<std::vector<double>> readNumericCSV(std::string filename,
			char comment = '#');
}
#endif // UTILITY_FUNCTIONS_H


