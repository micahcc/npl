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
 * @file mrimage.txx
 *
 *****************************************************************************/

#include <algorithm>
#include <cstring>
#include <typeinfo>
#include <Eigen/SVD>
#include <Eigen/Core>

#include "nifti.h"
#include "version.h"
#include "slicer.h"
#include "macros.h"
#include "ndarray.h"

namespace npl {

int writeNifti1Image(MRImage* out, gzFile file);
int writeNifti2Image(MRImage* out, gzFile file);

/**
 * @brief Constructor with initializer list
 *
 * @param a_args dimensions of input, the length of this initializer list
 * may not be fully used if a_args is longer than D. If it is shorter
 * then D then additional dimensions are left as size 1.
 */
template <size_t D,typename T>
MRImageStore<D,T>::MRImageStore(std::initializer_list<size_t> a_args) :
    NDArrayStore<D,T>(a_args), MRImage()
{
	orientDefault();
}

/**
 * @brief Constructor with vector
 *
 * @param a_args dimensions of input, the length of this initializer list
 * may not be fully used if a_args is longer than D. If it is shorter
 * then D then additional dimensions are left as size 1.
 */
template <size_t D,typename T>
MRImageStore<D,T>::MRImageStore(const std::vector<size_t>& dim) :
	NDArrayStore<D,T>(dim), MRImage()
{
    m_direction.resize(D,D);
    m_origin.resize(D);
    m_spacing.resize(D);
	orientDefault();
}

template <size_t D,typename T>
MRImageStore<D,T>::MRImageStore(size_t len, const size_t* size) :
	NDArrayStore<D,T>(len, size), MRImage()
{
    m_direction.resize(D,D);
    m_origin.resize(D);
    m_spacing.resize(D);
	orientDefault();
}

template <size_t D,typename T>
MRImageStore<D,T>::MRImageStore(size_t len, const size_t* size, T* ptr,
        const std::function<void(void*)>& deleter) :
	NDArrayStore<D,T>(len, size, ptr, deleter), MRImage()
{
    m_direction.resize(D,D);
    m_origin.resize(D);
    m_spacing.resize(D);
	orientDefault();
}


/**
 * @brief Default orientation (dir=ident, space=1 and origin=0), also resizes
 * them. So this could be called without first initializing size.
 */
void MRImage::orientDefault()
{
	for(size_t ii=0; ii<ndim(); ii++) {
		m_spacing[ii] = 1;
		m_origin[ii] = 0;
		for(size_t jj=0; jj<ndim(); jj++) 
			m_direction(ii,jj) = (ii==jj);
	}

    m_inv_direction = m_direction.inverse();
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
void MRImage::setOrient(const VectorXd& neworigin, 
        const VectorXd& newspace, const MatrixXd& newdir, bool reinit)
{

	if(reinit) 
		orientDefault();

	for(size_t ii=0; ii<newdir.rows() && ii < ndim(); ii++) {
		for(size_t jj=0; jj<newdir.cols() && jj< ndim(); jj++)
			m_direction(ii,jj) = newdir(ii,jj);
		origin(ii) = neworigin[ii];
		spacing(ii) = newspace[ii];
	}

    m_inv_direction = m_direction.inverse();
}

/**
 * @brief Returns reference to a value in the direction matrix.  
 * Each row indicates the direction of the grid in
 * RAS coordinates. This is the rotation of the Index grid. 
 *
 * @param row Row to access
 * @param col Column to access
 * 
 * @return Element in direction matrix
 */
double& MRImage::direction(int64_t row, int64_t col) 
{
    return m_direction(row, col);
}

/**
 * @brief Returns reference to a value in the direction matrix.  
 * Each row indicates the direction of the grid in
 * RAS coordinates. This is the rotation of the Index grid. 
 *
 * @param row Row to access
 * @param col Column to access
 * 
 * @return Element in direction matrix
 */
const double& MRImage::direction(int64_t row, int64_t col) const
{
    return m_direction(row, col);
}

/**
 * @brief Returns reference to the direction matrix.
 * Each row indicates the direction of the grid in
 * RAS coordinates. This is the rotation of the Index grid. 
 * 
 * @return Direction matrix
 */
const MatrixXd& MRImage::getDirection() const
{
    return m_direction;
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
void MRImage::setDirection(const MatrixXd& newdir, bool reinit)
{
	if(reinit) {
		for(size_t ii=0; ii < m_direction.rows(); ii++) {
			for(size_t jj=0; jj< m_direction.cols(); jj++) {
				m_direction(ii,jj) = (ii==jj);
			}
		}
	}

	for(size_t ii=0; ii<newdir.rows() && ii < m_direction.rows(); ii++) {
		for(size_t jj=0; jj<newdir.cols() && jj< m_direction.cols(); jj++)
			m_direction(ii,jj) = newdir(ii,jj);
	}

    m_inv_direction = m_direction.inverse();
}

/**
 * @brief Returns reference to a value in the origin vector. This is the
 * physical point that corresponds to index 0.
 *
 * @param row Row to access
 * 
 * @return Element in origin vector 
 */
double& MRImage::origin(int64_t row)
{
    return m_origin[row];
}

/**
 * @brief Returns reference to a value in the origin vector. This is the
 * physical point that corresponds to index 0.
 * 
 * @param row Row to access
 *
 * @return Element in origin vector 
 */
const double& MRImage::origin(int64_t row) const
{
    return m_origin[row];
}

/**
 * @brief Returns const reference to the origin vector. This is the physical
 * point that corresponds to index 0.
 * 
 * @return Origin vector 
 */
const VectorXd& MRImage::getOrigin() const
{
    return m_origin;
}

/**
 * @brief Sets the origin vector. This is the physical
 * point that corresponds to index 0. Note that min(current, new) elements 
 * will be copied
 *
 * @param neworigin the new origin vector to copy.
 * @param reinit Whether to reset everything to Identity/0 before applying.
 * You may want to do this if theinput matrices/vectors differ in dimension
 * from this image.
 */
void MRImage::setOrigin(const VectorXd& neworigin, bool reinit)
{
    if(reinit) {
        for(size_t ii=0; ii<m_origin.rows(); ii++)
            m_origin[ii] = 0;
    }

    for(size_t jj=0; jj<neworigin.rows() && jj< m_origin.rows(); jj++)
        origin(jj) = neworigin[jj];

}

/**
 * @brief Returns reference to a value in the spacing vector. This is the
 * physical distance between adjacent indexes. 
 * 
 * @param row Row to access
 * 
 * @return Element in spacing vector 
 */
double& MRImage::spacing(int64_t row)
{
    return m_spacing[row];
}

/**
 * @brief Returns reference to a value in the spacing vector. This is the
 * physical distance between adjacent indexes. 
 * 
 * @param row Row to access
 * 
 * @return Element in spacing vector 
 */
const double& MRImage::spacing(int64_t row) const
{
    return m_spacing[row];
}

/**
 * @brief Returns const reference to the spacing vector. This is the
 * physical distance between adjacent indexes. 
 * 
 * @return Spacing vector 
 */
const VectorXd& MRImage::getSpacing() const
{
    return m_spacing;
}

/**
 * @brief Sets the spacing vector. This is the physical
 * point that corresponds to index 0. Note that min(current, new) elements 
 * will be copied
 *
 * @param newspacing the new spacing vector to copy.
 * @param reinit Set the whole vector to 1s first. This might be useful if you
 * are setting fewer elements than dimensions
 * 
 */
void MRImage::setSpacing(const VectorXd& newspacing, bool reinit)
{
    if(reinit) {
        for(size_t ii=0; ii<m_spacing.rows(); ii++)
            m_spacing[ii] = 0;
    }

    for(size_t jj=0; jj<newspacing.rows() && jj< m_spacing.rows(); jj++)
        spacing(jj) = newspacing(jj);
}


/**
 * @brief Just dump image information
 */
template <size_t D,typename T>
void MRImageStore<D,T>::printSelf()
{
	std::cerr << D << "D Image\n[";
	for(size_t ii=0; ii<D; ii++)
		std::cerr << dim(ii) << ", ";
	std::cerr << "]\n";

	std::cerr << "Orientation:\nOrigin: [";
	for(size_t ii=0; ii<D; ii++)
		std::cerr << origin(ii) << ", ";
	std::cerr << "\nSpacing: [";
	for(size_t ii=0; ii<D; ii++)
		std::cerr << spacing(ii) << ", ";
	std::cerr << "\nDirection:\n";
	for(size_t ii=0; ii<D; ii++) {
		std::cerr << "[";
		for(size_t jj=0; jj<D; jj++)
			std::cerr << direction(ii,jj) << ", ";
		std::cerr << "]\n";
	}
}

#undef TYPEFUNC

/*******************************************************************************
 * Image Writers
 ******************************************************************************/

template <size_t D, typename T>
int MRImageStore<D,T>::write(std::string filename, double version) const
{
	std::string mode = "wb";
#if ZLIB_VERNUM >= 0x1280
	const size_t BSIZE = 1024*1024; //1M
#endif
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

#if ZLIB_VERNUM >= 0x1280
	gzbuffer(gz, BSIZE);
#endif

	if(nogz.substr(nogz.size()-4, 4) == ".nii") {
		if(version >= 2) {
			if(writeNifti2Image(gz) != 0) {
				std::cerr << "Error writing" << std::endl;
				gzclose(gz);
				return -1;
			}
		} else {
			if(writeNifti1Image(gz) != 0) {
				std::cerr << "Error writing" << std::endl;
				gzclose(gz);
				return -1;
			}
		}
    } else if(nogz.substr(nogz.size()-5, 5) == ".json") {
        writeJSON(gz);
	} else {
		std::cerr << "Unknown filetype: " << nogz.substr(nogz.rfind('.'))
			<< std::endl;
		gzclose(gz);
		return -1;
	}

	gzclose(gz);
	return 0;
}

template <size_t D, typename T>
int MRImageStore<D,T>::writeJSON(gzFile file) const
{
    ostringstream oss;
    oss << "{\n\"version\" : \"" << __version__<< "\",\n\"comment\" : \"supported "
        "type variables: uint8, int16, int32, float, cfloat, double, RGB, "
        "int8, uint16, uint32, int64, uint64, quad, cdouble, cquad, RGBA\",\n";
    oss << "\"type\": " << '"' << pixelTtoString(type()) << "\",\n";
    oss << "\"size\": [";
    for(size_t ii=0; ii<D; ii++) {
        if(ii) oss << ", ";
        oss << dim(ii);
    }
    oss << "],\n";

    oss << "\"spacing\": [";
    for(size_t ii=0; ii<D; ii++) {
        if(ii) oss << ", ";
        oss << spacing(ii);
    }
    oss << "],\n";

    oss << "\"origin\": [";
    for(size_t ii=0; ii<D; ii++) {
        if(ii) oss << ", ";
        oss << origin(ii);
    }
    oss << "],\n";
    
    oss << "\"direction\":\n[";;
    for(size_t ii=0; ii<D; ii++) {
        if(ii) oss << ",\n";
        oss << "[";
        for(size_t jj=0; jj<D; jj++) {
            if(jj) oss << ", ";
            oss << direction(ii, jj);
        }
        oss << "]";
    }
    oss << "],\n";
    
    int64_t index[D]; 
    oss << "\"values\" : ";
    for(NDConstIter<T> it(getConstPtr()); !it.eof(); ++it) {
        it.index(D, index);
        if(index[D-1] == 0)
            oss << "\n";
        for(int64_t dd=D-1; dd>=0; dd--) {
            if(index[dd] == 0) {
                oss << "[";
            } else {
                break;
            }
        }
        oss << *it;;
        for(int64_t dd=D-1; dd>=0; dd--) {
            if(index[dd] == dim(dd)-1) {
                oss << "]";
            } else {
                oss << ", ";
                break;
            }
        }
    }
    oss << "\n}\n";

    if(gzwrite(file, oss.str().c_str(), oss.str().length()) > 0)
        return 0;
	return -1;
}

template <size_t D, typename T>
int MRImageStore<D,T>::writeNifti1Image(gzFile file) const
{
	int ret = writeNifti1Header(file);
	if(ret != 0)
		return ret;
	ret = writePixels(file);
	return ret;
}

template <size_t D, typename T>
int MRImageStore<D,T>::writeNifti2Image(gzFile file) const
{
	int ret = writeNifti2Header(file);
	if(ret != 0)
		return ret;
	ret = writePixels(file);
	return ret;
}

template <size_t D, typename T>
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
	/***************************
	 * Orientation
	 **************************/
	// offsets
	if(D > 3)
		header.toffset = origin(3);
	for(size_t ii=0; ii<3 && ii<D; ii++)
		header.qoffset[ii] = origin(ii);

	for(size_t ii=0; ii<7 && ii<D; ii++)
		header.pixdim[ii] = spacing(ii);
	
	// Direction Matrix
	Matrix3d rotate;
	rotate.setIdentity();
	for(size_t rr=0; rr<3 && rr<D; rr++) {
		for(size_t cc=0; cc<3 && cc<D; cc++) {
			rotate(rr,cc) = direction(rr,cc);
		}
	}
	double det = rotate.determinant();
	if(fabs(det) < 1e-10) {
		std::cerr << "Non-invertible direction! Setting to identity\n"; 
		rotate.setIdentity();
	}

	// use SVD to orthogonalize, we basically just remove all scaling (make
	// eigenvalues 1)
	Eigen::JacobiSVD<Matrix3d> svd(rotate, Eigen::ComputeFullU|Eigen::ComputeFullV);
	rotate = svd.matrixU()*svd.matrixV().transpose();

	det = rotate.determinant();
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

template <size_t D, typename T>
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
		header.toffset = origin(3);
	for(size_t ii=0; ii<3 && ii<D; ii++)
		header.qoffset[ii] = origin(ii);

	for(size_t ii=0; ii<7 && ii<D; ii++)
		header.pixdim[ii] = spacing(ii);
	
	Matrix3d rotate;
	for(size_t rr=0; rr<3 && rr<D; rr++) {
		for(size_t cc=0; cc<3 && cc<D; cc++) {
			rotate(rr,cc) = direction(rr,cc);
		}
	}

	double det = rotate.determinant();
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

template <size_t D, typename T>
int MRImageStore<D,T>::writePixels(gzFile file) const
{
	// x is the fastest in nifti, for us it is the slowest
	std::vector<size_t> order;
	for(size_t ii=0 ; ii<ndim(); ii++)
		order.push_back(ii);

	Slicer it(ndim(), dim());
	it.setOrder(order);
	for(it.goBegin(); !it.isEnd(); ++it) {
		gzwrite(file, &this->_m_data[*it], sizeof(T));
	}
	return 0;
}

template <size_t D, typename T>
ptr<MRImage> MRImageStore<D,T>::cloneImage() const
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

	out->m_direction = m_direction;
	out->m_spacing   = m_spacing;
	out->m_origin    = m_origin;
	for(size_t ii=0; ii<D; ii++)
		out->m_units[ii] = m_units[ii];

	out->m_inv_direction = m_inv_direction;

	std::copy(this->_m_data, this->_m_data+total, out->_m_data);
	std::copy(this->_m_dim, this->_m_dim+D, out->_m_dim);

	return out;
}

/*****************************************************************************
 * Orientation Functions
 ****************************************************************************/

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
template <size_t D, typename T>
int MRImageStore<D,T>::indexToPoint(size_t len, const int64_t* index,
			double* rast) const
{
    Matrix<double, D, 1> vindex;
    Matrix<double, D, 1> vpoint;
    // copy in
    for(size_t ii=0; ii<len && ii<D; ii++) 
        vindex[ii] = index[ii];
    for(size_t ii=len; ii<D; ii++)
        vindex[ii] = 0;

    // apply transform
    // vpoint = m_direction*(vindex.array()*spacing.array())+origin;
    for(size_t rr = 0; rr<D; rr++) {
        vpoint[rr] = 0;
        for(size_t cc = 0; cc < D; cc++) 
            vpoint[rr] += m_direction(rr,cc)*vindex[cc]*spacing(cc);
        vpoint[rr] += origin(rr);
    }
    
    // copy out
    for(size_t ii=0; ii<len; ii++) 
        rast[ii] = vpoint[ii];
    for(size_t ii=len; ii<D; ii++) 
        rast[ii] = 0;
	
    return 0;
}

/**
 * @brief Converts a continous index to a RAS point index. Result in
 * may be outside the FOV. Input vector may be difference size that dimension.
 * Excess dimensions are ignored, missing dimensions are treated as zeros.
 *
 * @tparam D Dimension of image
 * @param len Length of xyz/ras arrays.
 * @param xyz Array in xyz... coordinates (maybe as long as you want).
 * @param ras Corresponding coordinate
 *
 * @return
 */
template <size_t D, typename T>
int MRImageStore<D,T>::indexToPoint(size_t len, const double* index,
			double* rast) const
{
    Matrix<double, D, 1> vindex;
    Matrix<double, D, 1> vpoint;
    // copy in
    for(size_t ii=0; ii<len && ii<D; ii++) 
        vindex[ii] = index[ii];
    for(size_t ii=len; ii<D; ii++)
        vindex[ii] = 0;

    // apply transform
    // vpoint = m_direction*(vindex.array()*spacing.array())+origin;
    for(size_t rr = 0; rr<D; rr++) {
        vpoint[rr] = 0;
        for(size_t cc = 0; cc < D; cc++) 
            vpoint[rr] += m_direction(rr,cc)*vindex[cc]*spacing(cc);
        vpoint[rr] += origin(rr);
    }
    
    // copy out
    for(size_t ii=0; ii<len; ii++) 
        rast[ii] = vpoint[ii];
    for(size_t ii=len; ii<D; ii++) 
        rast[ii] = 0;
	
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
template <size_t D, typename T>
int MRImageStore<D,T>::pointToIndex(size_t len, const double* rast,
			double* index) const
{
    Matrix<double, D, 1> vindex;
    Matrix<double, D, 1> vpoint;
    // copy in
    for(size_t ii=0; ii<len && ii<D; ii++) 
        vpoint[ii] = rast[ii];
    for(size_t ii=len; ii<D; ii++)
        vpoint[ii] = 0;

    // apply transform
    // vindex = (m_inv_direction*(vpoint-origin)).array()/spacing.array();
    for(size_t rr = 0; rr<D; rr++) {
        vindex[rr] = 0;
        for(size_t cc = 0; cc < D; cc++) 
            vindex[rr] += m_inv_direction(rr,cc)*(vpoint[cc]-origin(cc));

        vindex[rr] /= spacing(rr);
    }
    
    // copy out
    for(size_t ii=0; ii<len; ii++) 
        index[ii] = vindex[ii];
    for(size_t ii=len; ii<D; ii++) 
        index[ii] = 0;
	
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
template <size_t D, typename T>
int MRImageStore<D,T>::pointToIndex(size_t len, const double* rast,
			int64_t* index) const
{
    Matrix<double, D, 1> vindex;
    Matrix<double, D, 1> vpoint;
    // copy in
    for(size_t ii=0; ii<len && ii<D; ii++) 
        vpoint[ii] = rast[ii];
    for(size_t ii=len; ii<D; ii++)
        vpoint[ii] = 0;

    // apply transform
    // vindex = (m_inv_direction*(vpoint-origin)).array()/spacing.array();
    for(size_t rr = 0; rr<D; rr++) {
        vindex[rr] = 0;
        for(size_t cc = 0; cc < D; cc++) 
            vindex[rr] += m_inv_direction(rr,cc)*(vpoint[cc]-origin(cc));

        vindex[rr] /= spacing(rr);
    }
    
    // copy out
    for(size_t ii=0; ii<len; ii++) 
        index[ii] = round(vindex[ii]);
    for(size_t ii=len; ii<D; ii++) 
        index[ii] = 0;
	
    return 0;
}

/**
 * @brief Convert a vector in index coordinates to a vector in ras
 * coordinates. Vector is simply multiplied by the internal rotation
 * matrix.
 *
 * @param len Length of input vector (may be different than dimension -
 * extra values will be ignored, missing values will be assumed zero)
 * @param xyz Input vector in index space ijk.... 
 * @param ras Output vector in physical space. This is the product of the
 * input vector and rotation matrix 
 *
 * @return Success
 */
template <size_t D, typename T>
int MRImageStore<D,T>::orientVector(size_t len, const double* xyz, 
        double* ras) const
{
    Matrix<double, D, 1> vInd;
    Matrix<double, D, 1> vRAS;
    
    // copy in
    for(size_t ii=0; ii<len && ii<D; ii++) 
        vInd[ii] = xyz[ii];
    for(size_t ii=len; ii<D; ii++)
        vInd[ii] = 0;

    // apply transform
    // vpoint = m_direction*(vindex.array()*spacing.array())+origin;
    for(size_t rr = 0; rr<D; rr++) {
        vRAS[rr] = 0;
        for(size_t cc = 0; cc < D; cc++) 
            vRAS[rr] += m_direction(rr,cc)*vInd[cc]*spacing(cc);
    }
    
    // copy out
    for(size_t ii=0; ii<len; ii++) 
        ras[ii] = vRAS[ii];
    for(size_t ii=len; ii<D; ii++) 
        ras[ii] = 0;
	
    return 0;
}

/**
 * @brief Convert a vector in index coordinates to a vector in ras
 * coordinates. Vector is simply multiplied by the internal rotation
 * matrix.
 *
 * @param len Length of input vector (may be different than dimension -
 * extra values will be ignored, missing values will be assumed zero)
 * @param ras Input vector in physical space. 
 * @param xyz Output vector in index space ijk. This is the product of the
 * input vector and inverse rotation matrix 

 *
 * @return Success
 */
template <size_t D, typename T>
int MRImageStore<D,T>::disOrientVector(size_t len, const double* ras, 
        double* xyz) const
{
    Matrix<double, D, 1> vInd;
    Matrix<double, D, 1> vRAS;
    // copy in
    for(size_t ii=0; ii<len && ii<D; ii++) 
        vRAS[ii] = ras[ii];
    for(size_t ii=len; ii<D; ii++)
        vRAS[ii] = 0;

    // apply transform
    // vindex = (m_inv_direction*(vpoint-origin)).array()/spacing.array();
    for(size_t rr = 0; rr<D; rr++) {
        vInd[rr] = 0;
        for(size_t cc = 0; cc < D; cc++) 
            vInd[rr] += m_inv_direction(rr,cc)*vRAS[cc];

        vInd[rr] /= spacing(rr);
    }
    
    // copy out
    for(size_t ii=0; ii<len; ii++) 
        xyz[ii] = vInd[ii];
    for(size_t ii=len; ii<D; ii++) 
        xyz[ii] = 0;
	
    return 0;
}

/**
 * @brief Returns true if the point is within the field of view of the
 * image. Note, like all coordinates pass to MRImage, if the array given
 * differs from the dimensions of the image, then the result will either
 * pad out zeros and ignore extra values in the input array.
 *
 * @param len Length of RAS array
 * @param ras Array of Right-handed coordinates Right+, Anterior+, Superior+
 *
 * @return Whether the point would round to a voxel inside the image.
 */
template <size_t D, typename T>
bool MRImageStore<D,T>::pointInsideFOV(size_t len, const double* ras) const
{
    int64_t ind[D];
    pointToIndex(len, ras, ind);

	for(size_t ii=0; ii<D; ii++) {
		if(ind[ii] < 0 || ind[ii] >= this->_m_dim[ii])
			return false;
	}
	return true;
}

/**
 * @brief Returns true if the constinuous index is within the field of
 * view of the image. Note, like all coordinates pass to MRImage, if the
 * array given differs from the dimensions of the image, then the result
 * will either pad out zeros and ignore extra values in the input array.

 *
 * @param len Length of xyz array
 * @param xyz Array of continouos indices
 *
 * @return Whether the index would round to a voxel inside the image.
 */
template <size_t D, typename T>
bool MRImageStore<D,T>::indexInsideFOV(size_t len, const double* xyz) const
{
	for(size_t ii=0; ii<len && ii<D; ii++) {
		int64_t v = round(xyz[ii]);
		if(v < 0 || v >= this->_m_dim[ii])
			return false;
	}
	return true;
}

/**
 * @brief Returns true if the constinuous index is within the field of
 * view of the image. Note, like all coordinates pass to MRImage, if the
 * array given differs from the dimensions of the image, then the result
 * will either pad out zeros and ignore extra values in the input array.
 *
 * @param len Length of xyz array
 * @param xyz Array of indices
 *
 * @return Whether the index is inside the image
 */
template <size_t D, typename T>
bool MRImageStore<D,T>::indexInsideFOV(size_t len, const int64_t* xyz) const
{
	for(size_t ii=0; ii<len && ii<D; ii++) {
		int64_t v = xyz[ii];
		if(v < 0 || v >= this->_m_dim[ii])
			return false;
	}
	return true;
}

/*****************************************************************************
 * Copy Functions
 ****************************************************************************/

/**
 * @brief Performs a deep copy of the entire image and all metadata.
 *
 * @return Copied image.
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::copy() const
{
	return _copyCast(getConstPtr(), D, this->_m_dim, type());
}

/**
 * @brief Creates an identical array, but does not initialize pixel values.
 *
 * @return New array.
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::createAnother() const
{
	auto out = dPtrCast<MRImage>(createMRImage(D, dim(), type()));
	out->copyMetadata(getConstPtr());
	return out;
}

/**
 * @brief Create a new array that is the same underlying type as this.
 * If this is an image then it will also copy the metdata, but NOT the
 * pixels.
 *
 * @param newdims Number of dimensions in copied output
 * @param newsize Size of output, this array should be of size newdims
 * @param newtype Type of pixels in output array
 *
 * @return Image with identical orientation but different size and pixeltype
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::createAnother(size_t newdims,
        const size_t* newsize, PixelT newtype) const
{
	auto out = dPtrCast<MRImage>(createMRImage(newdims, newsize, newtype));
	out->copyMetadata(getConstPtr());
	return out;
}

/**
 * @brief Create a new array that is the same underlying type as this, but
 * with a different pixel type.
 *
 * @param newtype Type of pixels in output array
 *
 * @return Image with identical orientation and size but different pixel
 * type
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::createAnother(PixelT newtype) const
{
	auto out = dPtrCast<MRImage>(createMRImage(D, dim(), newtype));
	out->copyMetadata(getConstPtr());
	return out;
}

/**
 * @brief Create a new array that is the same underlying type as this,
 * and same pixel type and orientation as this, but with a different
 * size.
 *
 * @param newdims Number of dimensions in output array
 * @param newsize Input array of length newdims that gives the size of
 *                output array,
 *
 * @return Image with identical orientation and pixel type but different
 * size from this
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::createAnother(size_t newdims,
		const size_t* newsize) const
{
	auto out = dPtrCast<MRImage>(createMRImage(newdims, newsize, type()));
	out->copyMetadata(getConstPtr());
	return out;
}

/**
 * @brief Create a new image that is a copy of the input, possibly with new
 * dimensions and pixeltype. The new image will have all overlapping pixels
 * copied from the old image.
 *
 * This function just calls the outside copyCast, the reason for this
 * craziness is that making a template function nested in the already
 * huge number of templates I have kills the compiler, so we call an
 * outside function that calls templates that has all combinations of D,T.
 *
 * @param in Input image, anything that can be copied will be
 * @param newdims Number of dimensions in output image
 * @param newsize Size of output image
 * @param newtype Type of pixels in output image
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::copyCast(size_t newdims,
		const size_t* newsize, PixelT newtype) const
{
	return _copyCast(getConstPtr(), newdims, newsize, newtype);
}

/**
 * @brief Create a new image that is a copy of the input, with same dimensions
 * but pxiels cast to newtype. The new image will have all overlapping pixels
 * copied from the old image.
 *
 * This function just calls the outside copyCast, the reason for this
 * craziness is that making a template function nested in the already
 * huge number of templates I have kills the compiler, so we call an
 * outside function that calls templates that has all combinations of D,T.
 *
 * @param in Input image, anything that can be copied will be
 * @param newtype Type of pixels in output image
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::copyCast(PixelT newtype) const
{
	return _copyCast(getConstPtr(), newtype);
}

/**
 * @brief Create a new image that is a copy of the input, possibly with new
 * dimensions or size. The new image will have all overlapping pixels
 * copied from the old image. The new image will have the same pixel type as
 * the input image
 *
 * This function just calls the outside copyCast, the reason for this
 * craziness is that making a template function nested in the already
 * huge number of templates I have kills the compiler, so we call an
 * outside function that calls templates that has all combinations of D,T.
 *
 * @param in Input image, anything that can be copied will be
 * @param newdims Number of dimensions in output image
 * @param newsize Size of output image
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::copyCast(size_t newdims,
		const size_t* newsize) const
{
	return _copyCast(getConstPtr(), newdims, newsize);
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array. If len > ndim(), then the output may have more dimensions 
 * then the input, and in fact the extra dimensions may be larger than the
 * input image. If this happens, then data will still be extracted, but only
 * the overlapping segments of the new and old image will be copied. Also note
 * that index[] will not be accessed above ndim()
 *
 * @param len     Length of index/newsize arrays
 * @param index   ROI Index to start copying from.
 * @param size    ROI size. Note length 0 dimensions will be removed, while
 * length 1 dimensions will be left. 
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::extractCast(size_t len, const int64_t* index,
        const size_t* size) const
{
    return extractCast(len, index, size, type());
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array. If len > ndim(), then the output may have more dimensions 
 * then the input, and in fact the extra dimensions may be larger than the
 * input image. If this happens, then data will still be extracted, but only
 * the overlapping segments of the new and old image will be copied.
 *
 * @param len     Length of index/size arrays
 * @param index   Index to start copying from.
 * @param size Size of output image. Note length 0 dimensions will be
 * removed, while length 1 dimensions will be left. 
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::extractCast(size_t len, 
        const size_t* size) const
{
    return extractCast(len, NULL, size, type());
}

/**
 * @brief Copy a region of the input image. Note that the output must be 
 * smaller in each dimension, an have less than or equal to the number of input
 * dimensions.
 *
 * @param len     Length of index/size arrays
 * @param index   Index to start copying from.
 * @param size Size of output image. Note length 0 dimensions will be
 * removed, while length 1 dimensions will be left. 
 * @param newtype Pixel type of output image.
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::extractCast(size_t len, const int64_t* index,
        const size_t* size, PixelT newtype) const
{
    assert(size);
    assert(len < 10);
    
    int64_t ilower[D];
    int64_t iupper[D];
    
    size_t newdim = 0;
    size_t newsize[10];
    int64_t olower[10];
    int64_t oupper[10];

    // determine output size
    for(size_t dd=0; dd<len && dd<D; dd++) {
        if(size[dd] > 0) {
            newsize[newdim] = size[dd];
            olower[newdim] = 0;
            oupper[newdim] = size[dd]-1;
            newdim++;
        }
    }
    
    // create ROI in input image
    for(size_t dd=0; dd<D; dd++) {
        if(dd < len) {
            if(index)
                ilower[dd] = index[dd];
            else
                ilower[dd] = 0;

            if(size[dd] > 0) 
                iupper[dd] = ilower[dd]+size[dd]-1;
            else
                iupper[dd] = ilower[dd];
        } else {
            ilower[dd] = 0;
            iupper[dd] = 0;
        }

        if(iupper[dd] >= dim(dd)) {
            throw INVALID_ARGUMENT("Extracted Region is outside the input "
                    "image FOV");
        }
    }
    
    // create output
    auto out = dPtrCast<MRImage>(
			createMRImage(newdim, newsize, newtype));
    copyROI(getConstPtr(), ilower, iupper, out, olower, oupper, newtype);

	// copy spacing, origin and direction to out
	size_t odim1=0;
	for(size_t d1=0; d1<len && d1<D; d1++) {
		if(size[d1] > 0) {
			out->spacing(odim1) = spacing(d1);
			out->origin(odim1) = origin(d1);

			// second dimension, for direction
			size_t odim2=0;
			for(size_t d2=0; d2<len; d2++) {
				if(size[d2] > 0) {
					out->direction(odim1, odim2) = direction(d1, d2);
					odim2++;
				}
			}
			odim1++;
		}
	}

    return out;
}

/**
 * @brief Create a new array that is a copy of the input, possibly with new
 * dimensions or size. The new array will have all overlapping pixels
 * copied from the old array. The new array will have the same pixel type as
 * the input array. Index assumed to be [0,0,...], so the output image will 
 * start at the origin of this image.
 *
 * @param len     Length of index/size arrays
 * @param size Size of output image. Note length 0 dimensions will be
 * removed, while length 1 dimensions will be left. 
 * @param newtype Pixel type of output image.
 *
 * @return Image with overlapping sections cast and copied from 'in'
 */
template <size_t D, typename T>
ptr<NDArray> MRImageStore<D,T>::extractCast(size_t len, 
        const size_t* size, PixelT newtype) const
{
    return extractCast(len, NULL, size, newtype);
}


} //npl
