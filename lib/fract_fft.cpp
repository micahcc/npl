/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

/******************************************************************************
 * @file fract_fft.cpp
 * @brief Fractional fourier transform based on FFT
 ******************************************************************************/


/**
 * @brief Comptues the Fractional Fourier transform using FFTW for nlogn 
 * performance. 
 *
 * @param isize size of input/output
 * @param in Input array, may be the same as output, length sz
 * @param out Output array, may be the same as input, length sz
 * @param Buffer size
 * @param a Fraction, 1 = fourier transform, 3 = inverse fourier transform,
 * 4 = identity
 * @param buffer Buffer to do computations in, may be null, in which case new
 * memory will be allocated and deallocated during processing. Note that if
 * the provided buffer is not sufficient size a new buffer will be allocated
 * and deallocated, and a warning will be produced 
 * @param nonfft
 */
void fractional_fft(size_t sz, fftw_complex* in, fftw_complex* out, double a,
		size_t bsz = 0, fftw_complex* buffer = NULL, bool nonfft = false);

void fractional_fft(int64_t isize, int64_t usize, int64_t uppadsize,
		fftw_complex* inout, fftw_complex* buffer, double a,
		bool nonfft = false)
{
	const double PI = acos(-1);
	double phi = a*PI/2;
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / (usize*sqrt(sin(phi)));

	fftw_complex* upsampled = &buffer[uppadsize/4][0];
	fftw_complex* ab_chirp = &buffer[uppadsize][0];
	fftw_complex* b_chirp = &buffer[uppadsize*2][0];

	// upsampled version of input
	std::vector<complex<double>> upsampled(usize); // CACHE

	// create buffers and plans
	auto sigbuff = fftw_alloc_complex(uppadsize); // CACHE
	auto ab_chirp = createChirp(uppadsize, isize, (double)usize/(double)isize,
			alpha, beta, false); // CACHE
	auto b_chirp = createChirp(uppadsize, isize, (double)usize/(double)isize, 
			beta, 0, true); // CACHE

	fftw_plan sigbuff_plan_fwd = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan sigbuff_plan_rev = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff, 
			FFTW_BACKWARD, FFTW_MEASURE);

	// upsample input
	interp(input, upsampled);
	
	// pre-multiply 
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		complex<double> tmp1(ab_chirp[nn+uppadsize/2][0], 
				ab_chirp[nn+uppadsize/2][1]);
		upsampled[nn+usize/2] *= tmp1;;
	}

	// copy to padded buffer
	for(int64_t nn = -uppadsize/2; nn<=uppadsize/2; nn++) {
		if(nn <= usize/2 && nn >= -usize/2) {
			sigbuff[nn+uppadsize/2][0] = upsampled[nn+usize/2].real();
			sigbuff[nn+uppadsize/2][1] = upsampled[nn+usize/2].imag();
		} else {
			sigbuff[nn+uppadsize/2][0] = 0;
			sigbuff[nn+uppadsize/2][1] = 0;
		}
	}

	// convolve
	fftw_execute(sigbuff_plan_fwd);

	// not 100% clear on why sqrt works here, might be that the sqrt should be 
	// b_chirp fft
	double normfactor = sqrt(1./(uppadsize));
	for(size_t ii=0; ii<uppadsize; ii++) {
		complex<double> tmp1(sigbuff[ii][0], sigbuff[ii][1]);
		complex<double> tmp2(b_chirp[ii][0], b_chirp[ii][1]);
		tmp1 *= tmp2*normfactor;
		sigbuff[ii][0] = tmp1.real();
		sigbuff[ii][1] = tmp1.imag();
	}
	fftw_execute(sigbuff_plan_rev);

	// I am actually still not 100% on why these should be shifted up (-1) in 
	// sigbuff indexing
	// copy out, negatives
	for(int64_t ii=-usize/2; ii<=0; ii++) {
		upsampled[ii+usize/2].real(sigbuff[uppadsize-1+ii][0]);
		upsampled[ii+usize/2].imag(sigbuff[uppadsize-1+ii][1]);
	}
	// positives
	for(int64_t ii=1; ii<=usize/2; ii++) {
		upsampled[ii+usize/2].real(sigbuff[ii-1][0]);
		upsampled[ii+usize/2].imag(sigbuff[ii-1][1]);
	}

#ifdef DEBUG
	std::vector<double> mag;
	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) 
		mag[ii] = abs(upsampled[ii]);
	writePlot("fft_premult.tga", mag);
#endif //DEBUG

	// post-multiply
	for(int64_t ii=-usize/2; ii<=usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0], 
				ab_chirp[ii+uppadsize/2][1]);

		upsampled[ii+usize/2] *= tmp1*A_phi;
	}
	

	out.resize(input.size());
	interp(upsampled, out);

	fftw_free(sigbuff);
	fftw_free(b_chirp);
	fftw_free(ab_chirp);
	fftw_destroy_plan(sigbuff_plan_rev);
	fftw_destroy_plan(sigbuff_plan_fwd);
}

void fractional_ft(size_t isize, fftw_complex* in, fftw_complex* out, double a,
		size_t bsz = 0, fftw_complex* buffer = NULL, bool nonfft = false)
{
	double MINTHRESH = 0.000000000001;

	// bring a into range
	while(a < 0)
		a += 4;
	a = fmod(a, 4);
	
	// there are 3 sizes: isize: the original size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 4;
	int64_t uppadsize = round357(isize*approxratio); 
	int64_t usize;
	while( (usize = (uppadsize-1)/2) % 2 == 0) {
		uppadsize = round357(uppadsize+2);
	}

	// check/allocate buffer
	if(bsz < 4*uppadsize || !buffer) {
		bsz = 4*uppadsize;
		buffer = fftw_alloc_complex(bsz);
	}

	fftw_complex* current = &buffer[0][0];
	fftw_complex* padded = &buffer[uppadsize][0];
	fftw_complex* chirp_ab = &buffer[uppadsize*2][0];
	fftw_complex* chirp_b = &buffer[uppadsize*3][0];
	fftw_plan curr_to_out_fwd = fftw_plan_dft_1d(isize, current, out,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan curr_to_out_rev = fftw_plan_dft_1d(isize, current, out,
			FFTW_BACKWARD, FFTW_MEASURE);
	fftw_plan curr_to_curr_fwd = fftw_plan_dft_1d(isize, current, current,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan curr_to_curr_rev = fftw_plan_dft_1d(isize, current, current,
			FFTW_BACKWARD, FFTW_MEASURE);

	// copy input to buffer
	for(size_t ii=0; ii<isize; ii++) {
		current[ii][0] = in[ii][0];
		current[ii][1] = in[ii][1];
	}

	while(a != 0) {
		if(fabs(a-1) < MINTHRESH) {
			// fourier transform
			fftw_execute(curr_to_out_fwd);
			a = 0;
		} else if(fabs(a-2) < MINTHRESH) {
			// reverse
			for(size_t ii=0; ii<isize; ii++) {
				out[ii][0] = current[isize-1-ii][0];
				out[ii][1] = current[isize-1-ii][1];
			}
			a = 0;
		} else if(fabs(a-3) < MINTHRESH) {
			// inverse fourier transform
			fftw_execute(curr_to_out_rev);
			a = 0;
		} else if(a < 0.5) {
			// below range, do a FFT to bring into range of fractional_fft
			fftw_execute(curr_to_curr_fwd);
			a += 1;
		} else if(a <= 1.5) {
			// can do the general purpose fractional_fft
			fractional_fft();
			a = 0;
		} else if(a < 2.5) {
			// above range, do an rev FFT to bring into range of fractional_fft
			fftw_execute(curr_to_curr_rev);
			a -= 1;
		} else if(a < 3.5) {
			// way above range, do an signal reversal to bring into range of
			// fractional_fft
			// reverse
			for(size_t ii=0; ii<isize; ii++) {
				current[ii][0] = current[isize-1-ii][0];
				current[ii][1] = current[isize-1-ii][1];
			}
			a -= 2;
		} else {
			// 3.5-4, if we add 1 (subtract 3) we get to 0.5-1.5
			fftw_execute(curr_to_curr_rev);
			a -= 3;
		}
	}
}


void floatFrFFT(const std::vector<complex<double>>& input, float a_frac,
		vector<complex<double>>& out)

