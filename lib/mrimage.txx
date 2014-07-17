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

#include <algorithm>
#include <cstring>
#include <typeinfo>

#include "nifti.h"
#include "slicer.h"

namespace npl {

/**
 * @brief Constructor with initializer list
 *
 * @param a_args dimensions of input, the length of this initializer list
 * may not be fully used if a_args is longer than D. If it is shorter
 * then D then additional dimensions are left as size 1.
 */
template <int D,typename T>
MRImageStore<D,T>::MRImageStore(std::initializer_list<size_t> a_args) :
	NDArrayStore<D,T>(a_args), MRImage()
{
	orientDefault();
}

/**
 * @brief Constructor with array to set size
 *
 * @param dim dimensions of input 
 */
template <int D,typename T>
MRImageStore<D,T>::MRImageStore(const std::vector<size_t>& dim) : 
	NDArrayStore<D,T>(dim), MRImage()
{
	orientDefault();
}

/**
 * @brief Default orientation (dir=ident, space=1 and origin=0)
 */
template <int D,typename T>
void MRImageStore<D,T>::orientDefault()
{
	
	for(size_t ii=0; ii<D; ii++) {
		m_space[ii] = 1;
		m_origin[ii] = 0;
		for(size_t jj=0; jj<D; jj++) {
			m_dir(ii,jj) = (ii==jj);
		}
	}

	updateAffine();
}

/**
 * @brief Updates index->RAS affine transform cache
 */
template <int D,typename T>
void MRImageStore<D,T>::updateAffine()
{
	// first DxD section
	for(size_t ii=0; ii<D; ii++) {
		for(size_t jj=0; jj<D; jj++) {
			m_affine(ii,jj) = m_dir(ii, jj)*m_space[jj];
		}
	}
		
	// bottom row
	for(size_t jj=0; jj<D; jj++) 
		m_affine(D,jj) = 0;
	
	// last column
	for(size_t ii=0; ii<D; ii++) 
		m_affine(ii,D) = m_origin[ii];

	// bottom right
	m_affine(D,D) = 1;

	m_inv_affine = inverse(m_affine);
}


/**
* @brief Updates orientation information. If reinit is given then it will first
* set spacing to 1,1,1,1.... origin to 0,0,0,0... and direction to the identity.
* otherwise old values will be left. After this the first min(DIMENSION,dir.rows())
* columns and min(DIMENSION,dir.cols()) columns will be copies into the image 
* direction matrix. The first min(DIM,orig.rows()) and min(DIM,space.rows()) will
* be likewise copied. 
*
* @tparam D Image dimensionality
* @tparam T Pixeltype
* @param orig Input origin
* @param space Input spacing
* @param dir Input direction/rotation
* @param reinit Whether to reinitialize prior to copying
*/
template <int D,typename T>
void MRImageStore<D,T>::setOrient(const MatrixP& orig, const MatrixP& space, 
			const MatrixP& dir, bool reinit) 
{
	if(reinit) {
		orientDefault();
	}

	for(size_t ii=0; ii<dir.rows() && ii < D; ii++) {
		for(size_t jj=0; jj<dir.cols() && jj< D; jj++) 
			m_dir(ii,jj) = dir(ii,jj);
		m_origin[ii] = orig[ii];
		m_space[ii] = space[ii];
	}

	updateAffine();
}

/**
* @brief Updates spacing information. If reinit is given then it will first
* set spacing to 1,1,1,1.... otherwise old values will be left. The first
* min(DIM,space.rows()) will be copied. 
*
* @tparam D Image dimensionality
* @tparam T Pixeltype
* @param space Input spacing
* @param reinit Whether to reinitialize prior to copying
*/
template <int D,typename T>
void MRImageStore<D,T>::setSpacing(const MatrixP& space, bool reinit) 
{
	if(reinit) {
		for(size_t ii=0; ii<D; ii++)
			m_space[ii] = 1;
	}

	for(size_t ii=0; ii<space.rows() && ii < D; ii++) {
		m_space[ii] = space[ii];
	}

	updateAffine();
}

/**
* @brief Updates origin information. If reinit is given then it will first
* set origin to 0,0,0,0.... otherwise old values will be left. The first
* min(DIM,origin.rows()) will be copied. 
*
* @tparam D Image dimensionality
* @tparam T Pixeltype
* @param origin Input origin
* @param reinit Whether to reinitialize prior to copying
*/
template <int D,typename T>
void MRImageStore<D,T>::setOrigin(const MatrixP& origin, bool reinit) 
{
	if(reinit) {
		for(size_t ii=0; ii<D; ii++)
			m_origin[ii] = 1;
	}

	for(size_t ii=0; ii<origin.rows() && ii < D; ii++) {
		m_space[ii] = origin[ii];
	}

	updateAffine();
}

/**
* @brief Updates orientation information. If reinit is given then it will first
* set direction to the identity. otherwise old values will be left. After this
* the first min(DIMENSION,dir.rows()) columns and min(DIMENSION,dir.cols())
* columns will be copies into the image direction matrix. 
*
* @tparam D Image dimensionality
* @tparam T Pixeltype
* @param dir Input direction/rotation
* @param reinit Whether to reinitialize prior to copying
*/
template <int D,typename T>
void MRImageStore<D,T>::setDirection(const MatrixP& dir, bool reinit) 
{
	if(reinit) {
		for(size_t ii=0; ii < D; ii++) {
			for(size_t jj=0; jj< D; jj++) {
				m_dir(ii,jj) = (ii==jj);
			}
		}
	}

	for(size_t ii=0; ii<dir.rows() && ii < D; ii++) {
		for(size_t jj=0; jj<dir.cols() && jj< D; jj++) 
			m_dir(ii,jj) = dir(ii,jj);
	}

	updateAffine();
}


/**
 * @brief Just dump image information
 */
template <int D,typename T>
void MRImageStore<D,T>::printSelf()
{
	std::cerr << D << "D Image\n[";
	for(size_t ii=0; ii<D; ii++)
		std::cerr << dim(ii) << ", ";
	std::cerr << "]\n";

	std::cerr << "Orientation:\nOrigin: [";
	for(size_t ii=0; ii<D; ii++) 
		std::cerr << m_origin[ii] << ", ";
	std::cerr << "\nSpacing: [";
	for(size_t ii=0; ii<D; ii++) 
		std::cerr << m_space[ii] << ", ";
	std::cerr << "\nDirection:\n";
	for(size_t ii=0; ii<D; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D; jj++) 
			std::cerr << m_dir(ii,jj) << ", ";
		std::cerr << "]\n";
	}
	std::cerr << "\nAffine:\n";
	for(size_t ii=0; ii<D+1; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D+1; jj++) 
			std::cerr << m_affine(ii,jj) << ", ";
		std::cerr << "]\n";
	}
}

template <typename T>
shared_ptr<MRImage> createMRImageHelp(const std::vector<size_t>& dim)
{
	switch(dim.size()) {
		case 1:
			return std::make_shared<MRImageStore<1, T>>(dim);
		case 2:
			return std::make_shared<MRImageStore<2, T>>(dim);
		case 3:
			return std::make_shared<MRImageStore<3, T>>(dim);
		case 4:
			return std::make_shared<MRImageStore<4, T>>(dim);
		case 5:
			return std::make_shared<MRImageStore<5, T>>(dim);
		case 6:
			return std::make_shared<MRImageStore<6, T>>(dim);
		case 7:
			return std::make_shared<MRImageStore<7, T>>(dim);
		default:
			std::cerr << "Unsupported dimension: " << dim.size() << std::endl;
			return NULL;
	}

	return NULL;
}

shared_ptr<MRImage> createMRImage(const std::vector<size_t>& dim, PixelT ptype)
{
	switch(ptype) {
         case UINT8:
			return createMRImageHelp<uint8_t>(dim);
        break;
         case INT16:
			return createMRImageHelp<int16_t>(dim);
        break;
         case INT32:
			return createMRImageHelp<int32_t>(dim);
        break;
         case FLOAT32:
			return createMRImageHelp<float>(dim);
        break;
         case COMPLEX64:
			return createMRImageHelp<cfloat_t>(dim);
        break;
         case FLOAT64:
			return createMRImageHelp<double>(dim);
        break;
         case RGB24:
			return createMRImageHelp<rgb_t>(dim);
        break;
         case INT8:
			return createMRImageHelp<int8_t>(dim);
        break;
         case UINT16:
			return createMRImageHelp<uint16_t>(dim);
        break;
         case UINT32:
			return createMRImageHelp<uint32_t>(dim);
        break;
         case INT64:
			return createMRImageHelp<int64_t>(dim);
        break;
         case UINT64:
			return createMRImageHelp<uint64_t>(dim);
        break;
         case FLOAT128:
			return createMRImageHelp<long double>(dim);
        break;
         case COMPLEX128:
			return createMRImageHelp<cdouble_t>(dim);
        break;
         case COMPLEX256:
			return createMRImageHelp<cquad_t>(dim);
        break;
         case RGBA32:
			return createMRImageHelp<rgba_t>(dim);
        break;
		 default:
		return NULL;
	}
	return NULL;
}

#undef TYPEFUNC

/*******************************************************************************
 * Image Writers 
 ******************************************************************************/

template <int D, typename T>
int MRImageStore<D,T>::write(std::string filename, double version) const
{
	std::string mode = "wb";
	const size_t BSIZE = 1024*1024; //1M
	gzFile gz;

	// remove .gz to find the "real" format, 
	std::string nogz;
	if(filename.substr(filename.size()-3, 3) == ".gz") {
		nogz = filename.substr(0, filename.size()-3);
	} else {
		// if no .gz, then make encoding "transparent" (plain)
		nogz = filename;
		mode += 'T';
	}
	
	// go ahead and open
	gz = gzopen(filename.c_str(), mode.c_str());
	if(!gz) {
		std::cerr << "Could not open " << filename << " for writing!" << std::endl;
		return -1;
	}

	gzbuffer(gz, BSIZE);

	if(nogz.substr(nogz.size()-4, 4) == ".nii") {
		if(version >= 2) {
			std::cerr << "version >= 2" << std::endl;
			if(writeNifti2Image(gz) != 0) {
				std::cerr << "Error writing" << std::endl;
				gzclose(gz);
				return -1;
			}
		} else {
			std::cerr << "version < 2" << std::endl;
			if(writeNifti1Image(gz) != 0) {
				std::cerr << "Error writing" << std::endl;
				gzclose(gz);
				return -1;
			}
		}
	} else {
		std::cerr << "Unknown filetype: " << nogz.substr(nogz.rfind('.')) 
			<< std::endl;
		gzclose(gz);
		return -1;
	}

	gzclose(gz);
	return 0;
}

template <int D, typename T>
int MRImageStore<D,T>::writeNifti1Image(gzFile file) const
{
	std::cerr << "writeNifti1Image" << std::endl;
	int ret = writeNifti1Header(file);
#ifdef DEBUG
	std::cerr << "Writing Header" << std::endl;
#endif
	if(ret != 0) 
		return ret;
#ifdef DEBUG
	std::cerr << "Writing Pixels" << std::endl;
#endif
	ret = writePixels(file);
	return ret;
}

template <int D, typename T>
int MRImageStore<D,T>::writeNifti2Image(gzFile file) const
{
	std::cerr << "writeNifti2Image" << std::endl;
#ifdef DEBUG
	std::cerr << "Writing Header" << std::endl;
#endif
	int ret = writeNifti2Header(file);
	if(ret != 0) 
		return ret;
#ifdef DEBUG
	std::cerr << "Writing Pixels" << std::endl;
#endif
	ret = writePixels(file);
	return ret;
}

template <int D, typename T>
int MRImageStore<D,T>::writeNifti1Header(gzFile file) const
{
	static_assert(sizeof(nifti1_header) == 348, "Error, nifti header packing failed");
	nifti1_header header;
	std::fill((char*)&header, ((char*)&header)+sizeof(nifti1_header), 0);

	header.sizeof_hdr = 348;

	if(m_freqdim >= 0 && m_freqdim <= 2) 
		header.dim_info.bits.freqdim = m_freqdim+1; 
	if(m_phasedim>= 0 && m_phasedim<= 2) 
		header.dim_info.bits.phasedim = m_phasedim+1; 
	if(m_slicedim >= 0 && m_slicedim <= 2) 
		header.dim_info.bits.slicedim = m_slicedim+1; 

	// dimensions
	header.ndim = (short)ndim();
	for(size_t ii=0; ii<ndim(); ii++) {
		assert(dim(ii) <= SHRT_MAX);
		header.dim[ii] = (short)dim(ii);
	}

	header.datatype = type();
	header.bitpix = sizeof(T)*8;

	if(m_slice_order != 0) {
		assert(m_slice_start < SHRT_MAX);
		assert(m_slice_end < SHRT_MAX);

		header.slice_start = m_slice_start;
		header.slice_end = m_slice_end;
		header.slice_duration = m_slice_duration;
		header.slice_code = m_slice_order;
	}

	header.qform_code = 1;

	// orientation
	if(D > 3)
		header.toffset = m_origin[3];
	for(size_t ii=0; ii<3 && ii<D; ii++) 
		header.qoffset[ii] = m_origin[ii];

	for(size_t ii=0; ii<7 && ii<D; ii++) 
		header.pixdim[ii] = m_space[ii];
	
	Matrix<3,3> rotate; 
	for(size_t rr=0; rr<3 && rr<D; rr++) {
		for(size_t cc=0; cc<3 && cc<D; cc++) {
			rotate(rr,cc) = m_dir(rr,cc);
		}
	}

	double det = rotate.det();
	if(fabs(det)-1 > 0.0001) {
		std::cerr << "Non-orthogonal direction set! This may not end well" << std::endl;
	}

	if(det > 0) 
		header.qfac = 1;
	 else {
		header.qfac = -1;
		rotate(0,2) = -rotate(0,2);
		rotate(1,2) = -rotate(1,2);
		rotate(2,2) = -rotate(2,2);
	 }


	 double a = 0.5*sqrt(rotate(0,0)+rotate(1,1)+rotate(2,2)+1);
	 double b = 0.5*sqrt(rotate(0,0)-(rotate(1,1)+rotate(2,2))+1);
	 double c = 0.5*sqrt(rotate(1,1)-(rotate(0,0)+rotate(2,2))+1);
	 double d = 0.5*sqrt(rotate(2,2)-(rotate(0,0)+rotate(1,1))+1);

	 if(fabs(a) > 0.001) {
		 b = 0.25*(rotate(2,1)-rotate(1,2))/a;
		 c = 0.25*(rotate(0,2)-rotate(2,0))/a;
		 d = 0.25*(rotate(1,0)-rotate(0,1))/a;
	 } else if(fabs(b) > 0.001) {
		 c = 0.25*(rotate(0,1)+rotate(1,0))/b;
		 d = 0.25*(rotate(0,2)+rotate(2,0))/b;
		 a = 0.25*(rotate(2,1)-rotate(1,2))/b;
	 } else if(fabs(c) > 0.001) {
		 b = 0.25*(rotate(0,1)+rotate(1,0))/c ;
		 d = 0.25*(rotate(1,2)+rotate(2,1))/c ;
		 a = 0.25*(rotate(0,2)-rotate(2,0))/c ;
	 } else {
		 b = 0.25*(rotate(0,2)+rotate(2,0))/d ;
		 c = 0.25*(rotate(1,2)+rotate(2,1))/d ;
		 a = 0.25*(rotate(1,0)-rotate(0,1))/d ;
	 }

	 if(a < 0.0) {
		 b=-b;
		 c=-c;
		 d=-d;
//		 a=-a; 
	 }

	header.quatern[0] = b;
	header.quatern[1] = c;
	header.quatern[2] = d;
	
	//magic
	strncpy(header.magic,"n+1\0", 4);
	header.vox_offset = 352;


	// write over extension
	char ext[4] = {0,0,0,0};
	
	gzwrite(file, &header, sizeof(header));
	gzwrite(file, ext, sizeof(ext));

	return 0;
}

template <int D, typename T>
int MRImageStore<D,T>::writeNifti2Header(gzFile file) const
{
	static_assert(sizeof(nifti2_header) == 540, "Error, nifti header packing failed");
	nifti2_header header;
	std::fill((char*)&header, ((char*)&header)+sizeof(nifti2_header), 0);

	header.sizeof_hdr = 540;

	if(m_freqdim >= 0 && m_freqdim <= 2) 
		header.dim_info.bits.freqdim = m_freqdim+1; 
	if(m_phasedim>= 0 && m_phasedim<= 2) 
		header.dim_info.bits.phasedim = m_phasedim+1; 
	if(m_slicedim >= 0 && m_slicedim <= 2) 
		header.dim_info.bits.slicedim = m_slicedim+1;

	// dimensions
	header.ndim = ndim();
	for(size_t ii=0; ii<ndim(); ii++) {
		header.dim[ii] = dim(ii);
	}

	header.datatype = type();
	header.bitpix = sizeof(T)*8;

	if(m_slice_order != 0) {
		assert(m_slice_start < SHRT_MAX);
		assert(m_slice_end < SHRT_MAX);

		header.slice_start = m_slice_start;
		header.slice_end = m_slice_end;
		header.slice_duration = m_slice_duration;
		header.slice_code = m_slice_order;
	}

	header.qform_code = 1;

	// orientation
	if(D > 3)
		header.toffset = m_origin[3];
	for(size_t ii=0; ii<3 && ii<D; ii++) 
		header.qoffset[ii] = m_origin[ii];

	for(size_t ii=0; ii<7 && ii<D; ii++) 
		header.pixdim[ii] = m_space[ii];
	
	Matrix<3,3> rotate; 
	for(size_t rr=0; rr<3 && rr<D; rr++) {
		for(size_t cc=0; cc<3 && cc<D; cc++) {
			rotate(rr,cc) = m_dir(rr,cc);
		}
	}

	double det = rotate.det();
	if(fabs(det)-1 > 0.0001) {
		std::cerr << "Non-orthogonal direction set! This may not end well" << std::endl;
	}
	if(det > 0) 
		header.qfac = 1;
	 else {
		header.qfac = -1;
		rotate(0,2) = -rotate(0,2);
		rotate(1,2) = -rotate(1,2);
		rotate(2,2) = -rotate(2,2);
	 }


	 double a = 0.5*sqrt(rotate(0,0)+rotate(1,1)+rotate(2,2)+1);
	 double b = 0.5*sqrt(rotate(0,0)-(rotate(1,1)+rotate(2,2))+1);
	 double c = 0.5*sqrt(rotate(1,1)-(rotate(0,0)+rotate(2,2))+1);
	 double d = 0.5*sqrt(rotate(2,2)-(rotate(0,0)+rotate(1,1))+1);

	 if(fabs(a) > 0.001) {
		 b = 0.25*(rotate(2,1)-rotate(1,2))/a;
		 c = 0.25*(rotate(0,2)-rotate(2,0))/a;
		 d = 0.25*(rotate(1,0)-rotate(0,1))/a;
	 } else if(fabs(b) > 0.001) {
		 c = 0.25*(rotate(0,1)+rotate(1,0))/b;
		 d = 0.25*(rotate(0,2)+rotate(2,0))/b;
		 a = 0.25*(rotate(2,1)-rotate(1,2))/b;
	 } else if(fabs(c) > 0.001) {
		 b = 0.25*(rotate(0,1)+rotate(1,0))/c ;
		 d = 0.25*(rotate(1,2)+rotate(2,1))/c ;
		 a = 0.25*(rotate(0,2)-rotate(2,0))/c ;
	 } else {
		 b = 0.25*(rotate(0,2)+rotate(2,0))/d ;
		 c = 0.25*(rotate(1,2)+rotate(2,1))/d ;
		 a = 0.25*(rotate(1,0)-rotate(0,1))/d ;
	 }

	 if(a < 0.0) {
		 b=-b;
		 c=-c;
		 d=-d;
//		 a=-a; 
	 }

	header.quatern[0] = b;
	header.quatern[1] = c;
	header.quatern[2] = d;
	
	//magic
	strncpy(header.magic,"n+2\0", 4);
	header.vox_offset = 544;

	// write over extension
	char ext[4] = {0,0,0,0};
	
	gzwrite(file, &header, sizeof(header));
	gzwrite(file, ext, sizeof(ext));
	
	return 0;
}
//
//template<>
//int MRImage::writePixels<cfloat_t>(gzFile file) const
//{
//	// x is the fastest in nifti, for us it is the slowest
//	list<size_t> order;
//	for(size_t ii=0 ; ii<order.size(); ii++)
//		order.push_back(ii);
//
//	cdouble_t tmp;
//	for(auto it = cbegin_cdbl(order); !it.isEnd(); ++it) {
//		double re = it.get().real();
//		double im = it.get().imag();
//		gzwrite(file, &re, sizeof(double));
//		gzwrite(file, &im, sizeof(double));
//	}
//	return 0;
//}
//
//template<>
//int MRImage::writePixels<cdouble_t>(gzFile file) const
//{
//	// x is the fastest in nifti, for us it is the slowest
//	list<size_t> order;
//	for(size_t ii=0 ; ii<order.size(); ii++)
//		order.push_back(ii);
//
//	cdouble_t tmp;
//	for(auto it = cbegin_cdbl(order); !it.isEnd(); ++it) {
//		float re = it.get().real();
//		float im = it.get().imag();
//		gzwrite(file, &re, sizeof(float));
//		gzwrite(file, &im, sizeof(float));
//	}
//	return 0;
//}
//
//template <>
//int MRImage::writePixels<rgba_t>(gzFile file) const
//{
//
//	// x is the fastest in nifti, for us it is the slowest
//	list<size_t> order;
//	for(size_t ii=0 ; ii<order.size(); ii++)
//		order.push_back(ii);
//
//	for(auto it = cbegin_rgba(order); !it.isEnd(); ++it) {
//		char r = it.get().red;
//		char g = it.get().green;
//		char b = it.get().blue;
//		char a = it.get().alpha;
//		gzwrite(file, &r, sizeof(char));
//		gzwrite(file, &g, sizeof(char));
//		gzwrite(file, &b, sizeof(char));
//		gzwrite(file, &a, sizeof(char));
//	}
//	return 0;
//}

template <int D, typename T>
int MRImageStore<D,T>::writePixels(gzFile file) const
{
	// x is the fastest in nifti, for us it is the slowest
	std::list<size_t> order;
	for(size_t ii=0 ; ii<ndim(); ii++)
		order.push_back(ii);

	for(Slicer it(ndim(), dim(), order); !it.isEnd(); ++it) {
		gzwrite(file, &this->_m_data[*it], sizeof(T));
	}
	return 0;

}

template <int D, typename T>
std::shared_ptr<MRImage> MRImageStore<D,T>::cloneImage() const
{
	std::vector<size_t> newdims(this->_m_dim, this->_m_dim+D);
	auto out = std::make_shared<MRImageStore<D,T>>(newdims);

	out->m_slice_timing   = m_slice_timing;
	out->m_freqdim        = m_freqdim;
	out->m_phasedim       = m_phasedim;
	out->m_slicedim       = m_slicedim;
	out->m_slice_duration = m_slice_duration;
	out->m_slice_start    = m_slice_start;
	out->m_slice_end      = m_slice_end;
	out->m_slice_order    = m_slice_order;
	
	size_t total = 1;
	for(size_t ii=0; ii<D; ii++)
		total *= this->_m_dim[ii];

	out->m_dir       	 = m_dir;
	out->m_space     	 = m_space;
	out->m_origin    	 = m_origin;
	for(size_t ii=0; ii<D; ii++)
		out->m_units[ii] = m_units[ii];

	out->m_affine    	 = m_affine;
	out->m_inv_affine	 = m_inv_affine;

	std::copy(this->_m_data, this->_m_data+total, out->_m_data);
	std::copy(this->_m_dim, this->_m_dim+D, out->_m_dim);

	return out;
}
/**
 * @brief Converts a integer index to a RAS point index. Result in
 * may be outside the FOV. Input vector may be difference size that dimension.
 * Excess dimensions are ignored, missing dimensions are treated as zeros.
 *
 * @tparam D Dimension of image
 * @tparam T Pixeltype
 * @param index Index (may be out of bounds)
 * @param rast point in Right handed increasing RAS coordinate system
 *
 * @return 
 */
template <int D, typename T>
int MRImageStore<D,T>::indexToPoint(const std::vector<int64_t>& index,
		std::vector<double>& rast) const
{
	Matrix<D+1,1> in(index);
	in[D] = 1;
	Matrix<D+1,1> out;
	affine().mvproduct(in, out);
	rast.resize(D);
	for(size_t ii=0; ii<D; ii++)
		rast[ii] = out[ii];
	return 0;
}

/**
 * @brief Converts a continous index to a RAS point index. Result in
 * may be outside the FOV. Input vector may be difference size that dimension.
 * Excess dimensions are ignored, missing dimensions are treated as zeros.
 *
 * @tparam D Dimension of image
 * @tparam T Pixeltype
 * @param index Index (may be out of bounds)
 * @param rast point in Right handed increasing RAS coordinate system
 *
 * @return 
 */
template <int D, typename T>
int MRImageStore<D,T>::indexToPoint(const std::vector<double>& index,
		std::vector<double>& rast) const
{
	Matrix<D+1,1> in(index);
	in[D] = 1;
	Matrix<D+1,1> out;
	affine().mvproduct(in, out);
	rast.resize(D);
	for(size_t ii=0; ii<D; ii++)
		rast[ii] = out[ii];
	return 0;
}

/**
 * @brief Converts a point to an continous index. Could result in negative
 * values. Input vector may be difference size that dimension. Excess
 * dimensions are ignored, missing dimensions are treated as zeros.
 *
 * @tparam D Dimension of image
 * @tparam T Pixeltype
 * @param rast input in Right handed increasing RAS coordinate system
 * @param index Index (may be out of bounds)
 *
 * @return 
 */
template <int D, typename T>
int MRImageStore<D,T>::pointToIndex(const std::vector<double>& rast,
		std::vector<double>& index) const
{
	Matrix<D+1,1> in(rast);
	in[D] = 1;
	Matrix<D+1,1> out;
	iaffine().mvproduct(in, out);
	index.resize(D);
	for(size_t ii=0; ii<D; ii++)
		index[ii] = out[ii];
	return 0;
}

/**
 * @brief Converts a point to an int64 index. Could result in negative values.
 * Input vector may be difference size that dimension. Excess dimensions are 
 * ignored, missing dimensions are treated as zeros.
 *
 * @tparam D Dimension of image
 * @tparam T Pixeltype
 * @param rast input in Right handed increasing RAS coordinate system
 * @param index Index (may be out of bounds)
 *
 * @return 
 */
template <int D, typename T>
int MRImageStore<D,T>::pointToIndex(const std::vector<double>& rast,
		std::vector<int64_t>& index) const
{
	Matrix<D+1,1> in(rast);
	in[D] = 1;
	Matrix<D+1,1> out;
	iaffine().mvproduct(in, out);
	index.resize(D);
	for(size_t ii=0; ii<D; ii++) 
		index[ii] = round(out[ii]);
	return 0;
}

/* Linear Kernel Sampling */
double linKern(double x)
{
	return fabs(1-fmin(1,fabs(x)));
}


/**
 * @brief Linearly interpolates image at a point.
 *
 * @param point Continuous point location (could be outside FOV)
 * @param bound	What to return if the value is outside the FOV
 * @param outside Set to true if the value is outside, false otherwise
 *
 * @return 
 */
template <int D, typename T>
double MRImageStore<D,T>::linSamplePt(const std::vector<double>& point, 
		BoundaryConditionT bound, bool& outside)
{
	std::vector<double> cindex;
	pointToIndex(point, cindex);
	return linSampleInd(cindex, bound, outside);
}

/**
 * @brief Linearly interpolates image at a contiuous index
 *
 * @param cindex Continuous index location (could be outside FOV)
 * @param bound	What to return if the value is outside the FOV
 * @param outside Set to true if the value is outside, false otherwise. 
 * 			sometimes boundary points can trigger this if they are identically
 *			on the wall because their support might include outside voxels.
 *
 * @return 
 */
template <int D, typename T>
double MRImageStore<D,T>::linSampleInd(const std::vector<double>& incindex,
		BoundaryConditionT bound, bool& outside)
{
	double cindex[D];
	std::vector<int64_t> index(D, 0);
	
	// in case incindex is of the wrong size, copy
	for(size_t ii=0 ; ii<incindex.size() && ii < D; ii++)
		cindex[ii] = incindex[ii];
	for(size_t ii=incindex.size(); ii < D; ii++)
		cindex[ii] = 0;

	//kernels essentially 1D, so we can save time by combining 1D kernls
	//rather than recalculating
	const int kpoints = 2;
	double karray[D*kpoints];
	int64_t indarray[D*kpoints];
	for(int dd = 0; dd < D; dd++) {
		indarray[dd*kpoints+0] = floor(cindex[dd]);
		indarray[dd*kpoints+1] = indarray[dd*kpoints+0]+1;
		karray[dd*kpoints+0] = linKern(indarray[dd*kpoints+0]-cindex[dd]);
		karray[dd*kpoints+1] = linKern(indarray[dd*kpoints+1]-cindex[dd]);
	}

	bool iioutside = false;
	outside = false;
	double pixval = 0;
	double weight = 0;
	div_t result;
	//iterator over points in the neighborhood
	for(int ii = 0 ; ii < pow(kpoints, D); ii++) {
		weight = 1;
		
		//convert to local index, compute weight
		result.quot = ii;
		iioutside = false;
		for(int dd = 0; dd < D; dd++) {
			result = std::div(result.quot, kpoints);
			weight *= karray[dd*kpoints+result.rem];
			index[dd] = indarray[dd*kpoints+result.rem];
			iioutside = iioutside || index[dd] < 0 || index[dd] >= dim(dd);
		}

		outside = iioutside || outside;
	
		// just use clamped value
		if(iioutside) {
			if(bound == ZEROFLUX) {
				// clamp
				for(size_t dd=0; dd<D; dd++)
					index[dd] = clamp<int64_t>(0, dim(dd)-1, index[dd]);
			} else if(bound == WRAP) {
				// wrap
				for(size_t dd=0; dd<D; dd++) {
					if(index[dd] < 0)
						index[dd] = dim(dd)-(std::abs(index[dd])%dim(dd));
					else if(index[dd] >= dim(dd))
						index[dd] = index[dd]%dim(dd);
				}
			} else if(bound == CONSTZERO) {
				// set wieght to zero, then just clamp
				weight = 0;
				for(size_t dd=0; dd<D; dd++)
					index[dd] = clamp<int64_t>(0, dim(dd)-1, index[dd]);
			}
		} 

//		std::cerr << "Point: " << ii << " weight: " << weight << " Cont. Index: " ;
//		for(size_t ii=0; ii<D; ii++)
//			std::cerr << incindex[ii] << ",";
//		std::cerr << " Adj Cont Index: ";
//		for(size_t ii=0; ii<D; ii++)
//			std::cerr << cindex[ii] << ",";
//		std::cerr << ", Index: " ;
//		for(size_t ii=0; ii<D; ii++)
//			std::cerr << index[ii] << ",";
//		std::cerr << std::endl;

		pixval += weight*get_dbl(index);

	}

	return pixval;
}

/**
 * @brief Nearest Neigbor interpolation, handles outside values.
 *
 * @param point	Point to get
 * @param bound	What to return if the value is outside the FOV
 * @param outside Set to true if the value is outside, false otherwise
 *
 * @return 
 */
template <int D, typename T>
double MRImageStore<D,T>::nnSamplePt(const std::vector<double>& point,
		BoundaryConditionT bound, bool& outside)
{
	std::vector<double> cindex;
	pointToIndex(point, cindex);
	return nnSampleInd(cindex, bound, outside);
}

/**
 * @brief Nearest Neigbor interpolation, handles outside values.
 *
 * @param point	Point to get
 * @param bound	What to return if the value is outside the FOV
 * @param outside Set to true if the value is outside, false otherwise
 *
 * @return 
 */
template <int D, typename T>
double MRImageStore<D,T>::nnSampleInd(const std::vector<int64_t>& inindex,
		BoundaryConditionT bound, bool& outside)
{
	std::vector<int64_t> index(D, 0);
	
	// in case incindex is of the wrong size, copy
	for(size_t ii=0 ; ii<inindex.size() && ii < D; ii++)
		index[ii] = inindex[ii];
	for(size_t ii=inindex.size(); ii < D; ii++)
		index[ii] = 0;

	//convert to local index, compute weight
	outside = false;
	for(int dd = 0; dd < D; dd++) {
		outside = outside || index[dd] < 0 || index[dd] >= dim(dd);
	}

	// just use clamped value
	if(outside) {
		if(bound == ZEROFLUX) {
			// clamp
			for(size_t dd=0; dd<D; dd++)
				index[dd] = clamp<int64_t>(0, dim(dd)-1, index[dd]);
		} else if(bound == WRAP) {
			// wrap
			for(size_t dd=0; dd<D; dd++) {
				if(index[dd] < 0)
					index[dd] = dim(dd)-(std::abs(index[dd])%dim(dd));
				else if(index[dd] >= dim(dd))
					index[dd] = index[dd]%dim(dd);
			}
		} else if(bound == CONSTZERO) {
			return 0;
		}
	} 

	return get_dbl(index);
}

/**
 * @brief Nearest Neigbor interpolation, handles outside values.
 *
 * @param point	Point to get
 * @param bound	What to return if the value is outside the FOV
 * @param outside Set to true if the value is outside, false otherwise
 *
 * @return 
 */
template <int D, typename T>
double MRImageStore<D,T>::nnSampleInd(const std::vector<double>& incindex,
		BoundaryConditionT bound, bool& outside)
{
	std::vector<int64_t> index(incindex.size());
	for(size_t dd=0; dd<incindex.size(); dd++)
		index[dd] = round(incindex[dd]);

	return nnSampleInd(index, bound, outside);
}

} //npl
