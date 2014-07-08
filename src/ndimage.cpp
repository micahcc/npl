
#include "ndimage.h"
#include "ndimage.txx"

#include "ndarray.h"
#include "nifti.h"
#include "byteswap.h"
#include "slicer.h"

#include "zlib.h"

#include <cstring>

namespace npl {

/* Functions */
NDImage* readNiftiImage(gzFile file, bool verbose, double version);
int readNifti2Header(gzFile file, nifti2_header* header, bool* doswap, bool verbose);
int readNifti1Header(gzFile file, nifti1_header* header, bool* doswap, bool verbose);
int writeNifti1Image(NDImage* out, gzFile file);
int writeNifti2Image(NDImage* out, gzFile file);

template class NDImageStore<1, double>;
template class NDImageStore<1, float>;
template class NDImageStore<1, int64_t>;
template class NDImageStore<1, uint64_t>;
template class NDImageStore<1, int32_t>;
template class NDImageStore<1, uint32_t>;
template class NDImageStore<1, int16_t>;
template class NDImageStore<1, uint16_t>;
template class NDImageStore<1, int8_t>;
template class NDImageStore<1, uint8_t>;

template class NDImageStore<2, double>;
template class NDImageStore<2, float>;
template class NDImageStore<2, int64_t>;
template class NDImageStore<2, uint64_t>;
template class NDImageStore<2, int32_t>;
template class NDImageStore<2, uint32_t>;
template class NDImageStore<2, int16_t>;
template class NDImageStore<2, uint16_t>;
template class NDImageStore<2, int8_t>;
template class NDImageStore<2, uint8_t>;

template class NDImageStore<3, double>;
template class NDImageStore<3, float>;
template class NDImageStore<3, int64_t>;
template class NDImageStore<3, uint64_t>;
template class NDImageStore<3, int32_t>;
template class NDImageStore<3, uint32_t>;
template class NDImageStore<3, int16_t>;
template class NDImageStore<3, uint16_t>;
template class NDImageStore<3, int8_t>;
template class NDImageStore<3, uint8_t>;

template class NDImageStore<4, double>;
template class NDImageStore<4, float>;
template class NDImageStore<4, int64_t>;
template class NDImageStore<4, uint64_t>;
template class NDImageStore<4, int32_t>;
template class NDImageStore<4, uint32_t>;
template class NDImageStore<4, int16_t>;
template class NDImageStore<4, uint16_t>;
template class NDImageStore<4, int8_t>;
template class NDImageStore<4, uint8_t>;

template class NDImageStore<5, double>;
template class NDImageStore<5, float>;
template class NDImageStore<5, int64_t>;
template class NDImageStore<5, uint64_t>;
template class NDImageStore<5, int32_t>;
template class NDImageStore<5, uint32_t>;
template class NDImageStore<5, int16_t>;
template class NDImageStore<5, uint16_t>;
template class NDImageStore<5, int8_t>;
template class NDImageStore<5, uint8_t>;

template class NDImageStore<6, double>;
template class NDImageStore<6, float>;
template class NDImageStore<6, int64_t>;
template class NDImageStore<6, uint64_t>;
template class NDImageStore<6, int32_t>;
template class NDImageStore<6, uint32_t>;
template class NDImageStore<6, int16_t>;
template class NDImageStore<6, uint16_t>;
template class NDImageStore<6, int8_t>;
template class NDImageStore<6, uint8_t>;

template class NDImageStore<7, double>;
template class NDImageStore<7, float>;
template class NDImageStore<7, int64_t>;
template class NDImageStore<7, uint64_t>;
template class NDImageStore<7, int32_t>;
template class NDImageStore<7, uint32_t>;
template class NDImageStore<7, int16_t>;
template class NDImageStore<7, uint16_t>;
template class NDImageStore<7, int8_t>;
template class NDImageStore<7, uint8_t>;

template class NDImageStore<8, double>;
template class NDImageStore<8, float>;
template class NDImageStore<8, int64_t>;
template class NDImageStore<8, uint64_t>;
template class NDImageStore<8, int32_t>;
template class NDImageStore<8, uint32_t>;
template class NDImageStore<8, int16_t>;
template class NDImageStore<8, uint16_t>;
template class NDImageStore<8, int8_t>;
template class NDImageStore<8, uint8_t>;


/* Pre-Compile Certain Image Types */
NDImage* readNDImage(std::string filename, bool verbose)
{
	const size_t BSIZE = 1024*1024; //1M
	auto gz = gzopen(filename.c_str(), "rb");
	gzbuffer(gz, BSIZE);
	
	NDImage* out = NULL;

	if((out = readNiftiImage(gz, verbose, 1))) {
		gzclose(gz);
		return out;
	}

	if((out = readNiftiImage(gz, verbose, 2))) {
		gzclose(gz);
		return out;
	}

	
	return NULL;
}

int writeNDImage(NDImage* img, std::string fn, bool nifti2)
{
	return img->write(fn, nifti2 ? 2 : 1);
}

template <typename T>
NDImage* readPixels(gzFile file, size_t vox_offset, 
		std::vector<size_t> dim, size_t pixsize, bool doswap)
{
	// jump to voxel offset
	gzseek(file, vox_offset, SEEK_SET);

	/* 
	 * Create Slicer Object to iterate through image slices
	 */

	// dim 0 is the fastest in nifti images, so go in that order
	std::list<size_t> order(dim.size(), 0);
	for(size_t ii=0; ii<dim.size(); ii++) 
		order.push_back(ii);
	
	Slicer slicer(dim, order);

	T tmp(0);
	NDImage* out;

	// someday this all might be simplify by using NDImage* and the 
	// dbl or int64 functions, as long as we trust that the type is
	// going to be good enough to caputre the underlying pixle type
	switch(dim.size()) {
		case 1: {
			auto typed = new NDImageStore<1, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 2:{
			auto typed = new NDImageStore<2, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 3:{
			auto typed = new NDImageStore<2, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 4:{
			auto typed = new NDImageStore<4, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 5:{
			auto typed = new NDImageStore<5, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 6:{
			auto typed = new NDImageStore<6, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 7:{
			auto typed = new NDImageStore<7, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 8:{
			auto typed = new NDImageStore<8, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, pixsize);
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
	};

	return out;
}

int readNifti1Header(gzFile file, nifti1_header* header, bool* doswap, 
		bool verbose)
{
	// seek to 0
	gzseek(file, 0, SEEK_SET);

	static_assert(sizeof(nifti1_header) == 348, "Error, nifti header packing failed");

	// read header
	gzread(file, header, sizeof(nifti1_header));
	std::cerr << header->magic << std::endl;
	if(strncmp(header->magic, "n+1", 3)) {
		gzclearerr(file);
		gzrewind(file);
		return -1;
	}

	// byte swap
	int64_t npixel = 1;
	if(header->sizeof_hdr != 348) {
		*doswap = true;
		swap<int32_t>(&header->sizeof_hdr);
		swap<int16_t>(&header->ndim);
		for(size_t ii=0; ii<7; ii++)
			swap<int16_t>(&header->dim[ii]);
		swap<float>(&header->intent_p1);
		swap<float>(&header->intent_p2);
		swap<float>(&header->intent_p3);
		swap<int16_t>(&header->intent_code);
		swap<int16_t>(&header->datatype);
		swap<int16_t>(&header->bitpix);
		swap<int16_t>(&header->slice_start);
		swap<float>(&header->qfac);
		for(size_t ii=0; ii<7; ii++)
			swap<float>(&header->pixdim[ii]);
		swap<float>(&header->vox_offset);
		swap<float>(&header->scl_slope);
		swap<float>(&header->scl_inter);
		swap<int16_t>(&header->slice_end);
		swap<float>(&header->cal_max);
		swap<float>(&header->cal_min);
		swap<float>(&header->slice_duration);
		swap<float>(&header->toffset);

		for(int32_t ii=0; ii<header->ndim; ii++)
			npixel *= header->dim[ii];
	}
	
	if(verbose) {
		std::cerr << "sizeof_hdr:" << header->sizeof_hdr << std::endl;
		std::cerr << "ndim:" << header->ndim << std::endl;
		for(size_t ii=0; ii<7; ii++)
			std::cerr << "dim[" << ii << "]:" << header->dim[ii] << std::endl;
		for(size_t ii=0; ii<7; ii++)
			std::cerr << "pixdim[" << ii << "]:" << header->pixdim[ii] << std::endl;

		std::cerr << "intent_p1:" << header->intent_p1 << std::endl;
		std::cerr << "intent_p2:" << header->intent_p2 << std::endl;
		std::cerr << "intent_p3:" << header->intent_p3 << std::endl;
		std::cerr << "intent_code:" << header->intent_code << std::endl;
		std::cerr << "datatype:" << header->datatype << std::endl;
		std::cerr << "bitpix:" << header->bitpix << std::endl;
		std::cerr << "slice_start:" << header->slice_start << std::endl;
		std::cerr << "qfac:" << header->qfac << std::endl;
		std::cerr << "vox_offset:" << header->vox_offset << std::endl;
		std::cerr << "scl_slope:" << header->scl_slope << std::endl;
		std::cerr << "scl_inter:" << header->scl_inter << std::endl;
		std::cerr << "slice_end:" << header->slice_end << std::endl;
		std::cerr << "cal_max:" << header->cal_max << std::endl;
		std::cerr << "cal_min:" << header->cal_min << std::endl;
		std::cerr << "slice_duration:" << header->slice_duration << std::endl;
		std::cerr << "toffset:" << header->toffset << std::endl;
	}
	
	if(header->sizeof_hdr != 348) {
		std::cerr << "Malformed nifti input" << std::endl;
		return -1;
	}
	return 0;
}

/* 
 * Nifti Readers 
 */
NDImage* readNiftiImage(gzFile file, bool verbose, double version)
{
	bool doswap = false;
	int16_t datatype = 0;
	size_t start;
	std::vector<size_t> dim;
	size_t psize;
	int qform_code = 0;
	std::vector<double> pixdim;
	std::vector<double> offset;
	std::vector<double> quatern(3,0);
	double qfac;
	double slice_duration;
	size_t slice_code;
	size_t slice_start;
	size_t slice_end;
	size_t freqdim;
	size_t phasedim;
	size_t slicedim;

	if(version <= 1) {
		nifti1_header header;
		if(readNifti1Header(file, &header, &doswap, verbose) < 0)
			return NULL;

		start = header.vox_offset;
		dim.resize(header.ndim, 0);
		for(int64_t ii=0; ii<header.ndim && ii < 7; ii++)
			dim[ii] = header.dim[ii];
		psize = (header.bitpix >> 3);
		qform_code = header.qform_code;
		
		slice_code = header.slice_code;
		slice_duration = header.slice_duration;
		slice_start = header.slice_start;
		slice_end = header.slice_end;
		freqdim = (int)(header.dim_info.bits.freqdim)-1;
		phasedim = (int)(header.dim_info.bits.phasedim)-1;
		slicedim = (int)(header.dim_info.bits.slicedim)-1;

		// pixdim
		pixdim.resize(header.ndim, 0);
		for(int64_t ii=0; ii<header.ndim && ii < 7; ii++)
			pixdim[ii] = header.pixdim[ii];

		// offset
		offset.resize(header.ndim, 0);
		for(int64_t ii=0; ii<header.ndim && ii < 3; ii++)
			offset[ii] = header.qoffset[ii];
		if(header.ndim > 3)
			offset[3] = header.toffset;

		// quaternion
		for(int64_t ii=0; ii<3 && ii<header.ndim; ii++)
			quatern[ii] = header.quatern[ii];
		qfac = header.qfac;

	} else {
		nifti2_header header;
		if(readNifti2Header(file, &header, &doswap, verbose) < 0)
			return NULL;
		
		start = header.vox_offset;
		dim.resize(header.ndim, 0);
		for(int64_t ii=0; ii<header.ndim && ii < 7; ii++)
			dim[ii] = header.dim[ii];
		psize = (header.bitpix >> 3);
		qform_code = header.qform_code;
		
		slice_code = header.slice_code;
		slice_duration = header.slice_duration;
		slice_start = header.slice_start;
		slice_end = header.slice_end;
		freqdim = (int)(header.dim_info.bits.freqdim)-1;
		phasedim = (int)(header.dim_info.bits.phasedim)-1;
		slicedim = (int)(header.dim_info.bits.slicedim)-1;

		// pixdim
		pixdim.resize(header.ndim, 0);
		for(int64_t ii=0; ii<header.ndim && ii < 7; ii++)
			pixdim[ii] = header.pixdim[ii];
		
		// offset
		offset.resize(header.ndim, 0);
		for(int64_t ii=0; ii<header.ndim && ii < 3; ii++)
			offset[ii] = header.qoffset[ii];
		if(header.ndim > 3)
			offset[3] = header.toffset;
		
		// quaternion
		for(int64_t ii=0; ii<3 && ii<header.ndim; ii++)
			quatern[ii] = header.quatern[ii];
		qfac = header.qfac;
	}

	NDImage* out;

	// create image
	switch(datatype) {
		// 8 bit
		case NIFTI_TYPE_INT8:
			out = readPixels<int8_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT8:
			out = readPixels<uint8_t>(file, start, dim, psize, doswap);
		break;
		// 16  bit
		case NIFTI_TYPE_INT16:
			out = readPixels<int16_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT16:
			out = readPixels<uint16_t>(file, start, dim, psize, doswap);
		break;
		// 32 bit
		case NIFTI_TYPE_INT32:
			out = readPixels<int32_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT32:
			out = readPixels<uint32_t>(file, start, dim, psize, doswap);
		break;
		// 64 bit int 
		case NIFTI_TYPE_INT64:
			out = readPixels<int64_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_UINT64:
			out = readPixels<uint64_t>(file, start, dim, psize, doswap);
		break;
		// floats
		case NIFTI_TYPE_FLOAT32:
			out = readPixels<float>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_FLOAT64:
			out = readPixels<double>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_FLOAT128:
			out = readPixels<long double>(file, start, dim, psize, doswap);
		break;
		// RGB
		case NIFTI_TYPE_RGB24:
			out = readPixels<rgb_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_RGBA32:
			out = readPixels<rgba_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_COMPLEX256:
			out = readPixels<cquad_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_COMPLEX128:
			out = readPixels<cdouble_t>(file, start, dim, psize, doswap);
		break;
		case NIFTI_TYPE_COMPLEX64:
			out = readPixels<cfloat_t>(file, start, dim, psize, doswap);
		break;
	}

	if(!out)
		return NULL;

	/* 
	 * Now that we have an Image*, we can fill in the remaining values from 
	 * the header
	 */

	// figure out orientation
	if(qform_code > 0) {
		/*
		 * set spacing 
		 */
		for(size_t ii=0; ii<out->ndim(); ii++)
			out->space(ii) = pixdim[ii];
		
		/* 
		 * set origin 
		 */
		// x,y,z
		for(size_t ii=0; ii<out->ndim(); ii++)
			out->origin(ii) = offset[ii];
		
		/* Copy Quaternions, and Make Rotation Matrix */

//		// copy quaternions, (we'll set a from R)
//		out->use_quaterns = true;
//		for(size_t ii=0; ii<3 && ii<out->ndim(); ii++)
//			out->quaterns[ii+1] = quatern[ii];
		
		// calculate a, copy others
		double b = quatern[0];
		double c = quatern[1];
		double d = quatern[2];
		double a = sqrt(1.0-(b*b+c*c+d*d));

		// calculate R, (was already identity)
		out->direction(0, 0) = a*a+b*b-c*c-d*d;

		if(out->ndim() > 1) {
			out->direction(0,1) = 2*b*c-2*a*d;
			out->direction(1,0) = out->direction(0, 1);
			out->direction(1,1) = a*a+c*c-b*b-d*d;
		}
		
		if(out->ndim() > 2) {
			out->direction(0,2) = 2*b*d+2*a*c;
			out->direction(1,2) = 2*c*d-2*a*b;
			out->direction(2,2) = a*a+d*d-c*c-b*b;
			
			out->direction(2,0) = out->direction(0,2);
			out->direction(1,2) = out->direction(2,1);
		}

		// finally update affine, but scale pixdim[z] by qfac temporarily
		if(qfac == -1 && out->ndim() > 2)
			out->space(2) = -out->space(2);
		out->updateAffine();
		if(qfac == -1 && out->ndim() > 2)
			out->space(2) = -out->space(2);
//	} else if(header.sform_code > 0) {
//		/* use the sform, since no qform exists */
//
//		// origin, last column
//		double di = 0, dj = 0, dk = 0;
//		for(size_t ii=0; ii<3 && ii<out->ndim(); ii++) {
//			di += pow(header.srow[4*ii+0],2); //column 0
//			dj += pow(header.srow[4*jj+1],2); //column 1
//			dk += pow(header.srow[4*kk+2],2); //column 2
//			out->origin(ii) = header.srow[4*ii+3]; //column 3
//		}
//		
//		// set direction and spacing
//		out->m_spacing[0] = sqrt(di);
//		out->m_dir[0*out->ndim()+0] = header.srow[4*0+0]/di;
//
//		if(out->ndim() > 1) {
//			out->m_spacing[1] = sqrt(dj);
//			out->m_dir[0*out->ndim()+1] = header.srow[4*0+1]/dj;
//			out->m_dir[1*out->ndim()+1] = header.srow[4*1+1]/dj;
//			out->m_dir[1*out->ndim()+0] = header.srow[4*1+0]/di;
//		}
//		if(out->ndim() > 2) {
//			out->m_spacing[2] = sqrt(dk);
//			out->m_dir[0*out->ndim()+2] = header.srow[4*0+2]/dk;
//			out->m_dir[1*out->ndim()+2] = header.srow[4*1+2]/dk;
//			out->m_dir[2*out->ndim()+2] = header.srow[4*2+2]/dk;
//			out->m_dir[2*out->ndim()+1] = header.srow[4*2+1]/dj;
//			out->m_dir[2*out->ndim()+0] = header.srow[4*2+0]/di;
//		}
//
//		// affine matrix
//		updateAffine();
	} else {
		// only spacing changes
		for(size_t ii=0; ii<dim.size(); ii++)
			out->space(ii) = pixdim[ii];
		out->updateAffine();
	}

	/************************************************************************** 
	 * Medical Imaging Varaibles Variables 
	 **************************************************************************/
	
	// direct copies
	out->m_slice_duration = slice_duration;
	out->m_slice_start = slice_start;
	out->m_slice_end = slice_end;
	out->m_freqdim = freqdim;
	out->m_phasedim = phasedim;
	out->m_slicedim = slicedim;

	if(out->m_slicedim > 0) {
		out->m_slice_timing.resize(out->dim(out->m_slicedim), NAN);
	}
			
	// we use the same encoding as nifti

	// slice timing
	switch(slice_code) {
		case NIFTI_SLICE_SEQ_INC:
			out->m_slice_order = SEQ;
			for(int ii=out->m_slice_start; ii<=out->m_slice_end; ii++)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_SEQ_DEC:
			out->m_slice_order = RSEQ;
			for(int ii=out->m_slice_end; ii>=out->m_slice_start; ii--)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_INC:
			out->m_slice_order = ALT;
			for(int ii=out->m_slice_start; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_start+1; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_DEC:
			out->m_slice_order = RALT;
			for(int ii=out->m_slice_end; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_end-1; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_INC2:
			out->m_slice_order = ALT_SHFT;
			for(int ii=out->m_slice_start+1; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_start; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_DEC2:
			out->m_slice_order = RALT_SHFT;
			for(int ii=out->m_slice_end-1; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_end; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
	}

	return out;
}

int readNifti2Header(gzFile file, nifti2_header* header, bool* doswap, 
		bool verbose)
{
	// seek to 0
	gzseek(file, 0, SEEK_SET);

	static_assert(sizeof(nifti2_header) == 540, "Error, nifti header packing failed");

	// read header
	gzread(file, header, sizeof(nifti2_header));
	std::cerr << header->magic << std::endl;
	if(strncmp(header->magic, "n+2", 3)) {
		gzclearerr(file);
		gzrewind(file);
		return -1;
	}

	// byte swap
	int64_t npixel = 1;
	if(header->sizeof_hdr != 540) {
		*doswap = true;
		swap(&header->sizeof_hdr);
		swap(&header->ndim);
		for(size_t ii=0; ii<7; ii++)
			swap(&header->dim[ii]);
		swap(&header->intent_p1);
		swap(&header->intent_p2);
		swap(&header->intent_p3);
		swap(&header->intent_code);
		swap(&header->datatype);
		swap(&header->bitpix);
		swap(&header->slice_start);
		swap(&header->qfac);
		for(size_t ii=0; ii<7; ii++)
			swap(&header->pixdim[ii]);
		swap(&header->vox_offset);
		swap(&header->scl_slope);
		swap(&header->scl_inter);
		swap(&header->slice_end);
		swap(&header->cal_max);
		swap(&header->cal_min);
		swap(&header->slice_duration);
		swap(&header->toffset);

		for(int32_t ii=0; ii<header->ndim; ii++)
			npixel *= header->dim[ii];
	}
	
	if(verbose) {
		std::cerr << "sizeof_hdr:" << header->sizeof_hdr << std::endl;
		std::cerr << "ndim:" << header->ndim << std::endl;
		for(size_t ii=0; ii<7; ii++)
			std::cerr << "dim[" << ii << "]:" << header->dim[ii] << std::endl;
		for(size_t ii=0; ii<7; ii++)
			std::cerr << "pixdim[" << ii << "]:" << header->pixdim[ii] << std::endl;

		std::cerr << "intent_p1:" << header->intent_p1 << std::endl;
		std::cerr << "intent_p2:" << header->intent_p2 << std::endl;
		std::cerr << "intent_p3:" << header->intent_p3 << std::endl;
		std::cerr << "intent_code:" << header->intent_code << std::endl;
		std::cerr << "datatype:" << header->datatype << std::endl;
		std::cerr << "bitpix:" << header->bitpix << std::endl;
		std::cerr << "slice_start:" << header->slice_start << std::endl;
		std::cerr << "qfac:" << header->qfac << std::endl;
		std::cerr << "vox_offset:" << header->vox_offset << std::endl;
		std::cerr << "scl_slope:" << header->scl_slope << std::endl;
		std::cerr << "scl_inter:" << header->scl_inter << std::endl;
		std::cerr << "slice_end:" << header->slice_end << std::endl;
		std::cerr << "cal_max:" << header->cal_max << std::endl;
		std::cerr << "cal_min:" << header->cal_min << std::endl;
		std::cerr << "slice_duration:" << header->slice_duration << std::endl;
		std::cerr << "toffset:" << header->toffset << std::endl;
	}
	
	if(header->sizeof_hdr != 540) {
		std::cerr << "Malformed nifti input" << std::endl;
		return -1;
	}
	return 0;
}

} // npl
