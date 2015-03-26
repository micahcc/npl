/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file utility.h File with random utilities, for string processing, for
 * determining basic filename information (basename/dirname).
 *
 *****************************************************************************/
#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <string>
#include <cmath>
#include <list>
#include <vector>

#define __FUNCTION_STR__ std::string(__PRETTY_FUNCTION__)

namespace npl {

/**
 * @brief Returns the directory name for the given file
 *
 * @param path Input path
 *
 * @return 				Output directory name of input path
 */
std::string dirname(std::string path);

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
 * @param delim Output: Delimiter found (this gets set).
 * @param comment lines with this first non-white space character will be ignored
 *
 * @return out vector of rows (stored in vectors)
 */
std::vector<std::vector<std::string>> readStrCSV(std::string filename,
        char& delim, char comment = '#');

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

/**
 * @brief Gamma distribution, used by Cannonical HRF from SPM
 *
 * @param x			Position
 * @param k 		Shape parameter
 * @param theta 	Scale parameter
 *
 * @return
 */
double gammaPDF(double x, double k, double theta);


/**
 * @brief Cannonical hemodynamic response funciton from SPM.
 * delay of response (relative to onset) = 6
 * delay of undershoot (relative to onset) = 16
 * dispersion of response = 1
 * dispersion of undershoot = 1
 * ratio of response to undershoot = 6
 * onset (seconds) = 0
 * length of kernel (seconds) = 32
 *
 * @param t 		Time of sampled value
 * @param rdelay 	Delay of onset (default 6)
 * @param udelay	Delay of undershot (relative to onset, default 16)
 * @param rdisp		Dispersion of response (default 1)
 * @param udisp		Dispersion of undershoot (default 1)
 * @param puRatio	Ratio of response to undershoot (default 6)
 * @param onset		Onset time (default 0)
 * @param total		Kernel time (default 32)
 *
 * @return 			Weight at the given time
 */
double cannonHrf(double t, double rdelay, double udelay, double rdisp,
		double udisp, double puRatio, double onset, double total);

/**
 * @brief The cannoical HRF with all default parameters
 *
 * @param t	Time from stimulus
 *
 * @return 	Response level
 */
double cannonHrf(double t);

/**
 * @brief Takes a 1 or 3 column format of regressor and produces
 * a timeseries sampeled every tr, starting at time t0, and running
 * for ntimes.
 *
 * 3 column format:
 *
 * onset duration value
 *
 * 1 column format (value of 1 is assumed):
 *
 * onsert
 *
 * @param spec
 * @param tr
 * @param ntimes
 * @param t0
 *
 * @return
 */
std::vector<double> getRegressor(std::vector<std::vector<double>>& spec,
		double tr, size_t ntimes, double t0);

/**
 * @brief Takes a 1 or 3 column format of regressor and produces
 * a timeseries sampeled every tr, starting at time t0, and running
 * for ntimes.
 *
 * 3 column format:
 *
 * onset duration value
 *
 * 1 column format (value of 1 is assumed):
 *
 * onsert
 *
 * @param spec
 * @param tr
 * @param ntimes
 * @param t0
 *
 * @return
 */
std::vector<double> getRegressor(std::vector<std::vector<double>>& spec,
		double tr, size_t ntimes, double t0);

/**
 * @brief In place simulation of BOL timeseries using the balloon model.
 * Habituation is accomplished by subtracting exponential moving average of
 * u, thus if u tends to be on a lot, the effective u will be lower, if it
 * turns on for the first time in a long time, the spike will be greater.
 * Learn is the weight of the current point, the previous moving average
 * value is weighted (1-learn)
 *
 * @param len Length of input/output buffer
 * @param iobuff Input/Output Buffer of values (input stimulus, output signal)
 * @param dt Timestep for simulation
 * @param learn The weight of the current point, the previous moving average
 * value is weighted (1-learn), setting learn to 1 removes all habituation
 * This limits the effect of habituation.
 */
void boldsim(size_t len, double* iobuff, double dt, double learn);

/**
 * @brief Convolves a signal and a function using loops (not fast)
 * No wrapping is done.
 *
 * for ii in 0...N-1
 * for jj in 0...M-1
 * if jj-ii valid:
 *     s[ii] = s[ii]*kern[jj-ii]
 *
 * @param signal Input signal sampled every (tr)
 * @param foo Function to sample backwards
 * @param tr Sampling time of input signal
 * @param length Number of samples to draw from foo
 */
void convolve(std::vector<double>& signal, double(*foo)(double),
		double tr, double length);

/**
 * @brief Computes the standard deviation from FWHM (because I can never
 * remember the ratio)
 *
 * @param fwhm  Full width-half max that we want to convert to equivalent
 * standard deviation
 *
 * @return  Standard deviation for a gaussian with the specified FWHM
 */
inline
double fwhm_to_sd(double fwhm)
{
  return fwhm/(2*sqrt(2*log(2)));
}

/**
 * @brief Computes the FWHM from from standard deviation (because I can never
 * remember the ratio)
 *
 * @param sd Standard deviation of a Gaussian function.
 *
 * @return Full-width-half-max size of the same gaussian function
 */
inline
double sd_to_fwhm(double sd)
{
  return sd*(2*sqrt(2*log(2)));
}

/**
 * @brief Memory map class. The basic gyst is that this works like a malloc
 * except that data may be initialized by file contents or left empty. This
 * is useful for matrix manipulation where files need to be opened and closed
 * a good bit.
 */
class MemMap
{
public:
	/**
	 * @brief Create a new mmap class by opening the specified file
	 * with the specified size. This version ALWAYS CREATES A NEW FILE.
	 * THIS WILL OVERWRITE THE OLD FILE
	 *
	 * @param fn Open the specified file for reading and writing.
	 * @param bsize Make the file size this number of bytes.
	 */
	MemMap(std::string fn, size_t bsize);

	/**
	 * @brief Open an example file as a memory map.
	 *
	 * @param fn Open the specified file for reading and writing.
	 * @param writeable Whether the opened file will be writeable.
	 * Note that no two processes should have the same file open as writeable
	 * at a time.
	 */
	MemMap(std::string fn, bool writeable=false);

	/**
	 * @brief Just create the object. Until open() is called the size() will
	 * be 0 and returned data() will be NULL;
	 */
	MemMap() : m_size(0), m_fd(0), m_data(NULL) {};

	/**
	 * @brief Open the specified file with the specified size. If createNew is
	 * set then a new file will be created an existing file will be deleted (if
	 * permissions allow it). This is always opened as writeable
	 *
	 * @param fn Open the specified file for reading and writing.
	 * @param bsize Make the file size this number of bytes. If createNew is
	 * false then this must match the current file size
	 * @return size of memory map
	 */
	int64_t openNew(std::string fn, size_t bsize);

	/**
	 * @brief Open the specified file with the specified size. If createNew is
	 * set then a new file will be created an existing file will be deleted (if
	 * permissions allow it).
	 *
	 * @param fn Open the specified file for reading.
	 * @param writeable whether the opened file will be writeable (note that
	 * no two processes should open the same file as writeable)
	 * @param quiet Whether to print errors when we can't open a file
	 * @return size of memory map
	 */
	int64_t openExisting(std::string fn, bool writeable, bool quiet = true);

	/**
	 * @brief Return true if a file is currently open
	 */
	bool isopen() { return m_size > 0; };

	/**
	 * @brief Close the current file (if one is open). data() will return NULL
	 * and size() will return 0 after this until open() is called again.
	 */
	void close();

	/**
	 * @brief Destructor, close file.
	 */
	~MemMap();

	/**
	 * @brief Get pointer to data.
	 *
	 * @return pointer to current data
	 */
	void* data() { return m_data; };

	/**
	 * @brief Get constant pointer to current data
	 *
	 * @return pointer to current data
	 */
	const void* data() const { return m_data; };

	/**
	 * @brief Length (in bytes) of memory map
	 *
	 * 0 not error, but not open
	 * >0 open
	 * -1 error
	 *
	 * @return Tells what the current state of the memory map is
	 */
	int64_t size() { return m_size; };
private:

	int64_t m_size;
	int m_fd;
	void* m_data;
};

}
#endif // UTILITY_FUNCTIONS_H


