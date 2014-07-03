
#include "ndimage.h"
#include "ndarray.h"
#include "nifti.h"
#include "byteswap.h"
#include "slicer.h"

#include "zlib.h"

#include <cstring>

/* Functions */
NDImage* readNDImage(std::string filename);
void writeNDImage(NDImage* img, std::string filename);
NDImage* readNifti1Image(gzFile file);
NDImage* readNifti2Image(gzFile file);
int writeNifti1Image(NDImage* out, gzFile file);
int writeNifti2Image(NDImage* out, gzFile file);

template class NDImageStore<1, int8_t>;
template class NDImageStore<2, int8_t>;
template class NDImageStore<3, int8_t>;
template class NDImageStore<4, int8_t>;
template class NDImageStore<5, int8_t>;
template class NDImageStore<6, int8_t>;
template class NDImageStore<7, int8_t>;
template class NDImageStore<8, int8_t>;

/* Pre-Compile Certain Image Types */
NDImage* readNDImage(std::string filename)
{
	const size_t BSIZE = 1024*1024; //1M
	auto gz = gzopen(filename.c_str(), "rb");
	gzbuffer(gz, BSIZE);
	
	NDImage* out = NULL;

	if((out = readNifti1Image(gz))) {
		gzclose(gz);
		return out;
	}

	if((out = readNifti2Image(gz))) {
		gzclose(gz);
		return out;
	}

	
	return NULL;
}

int writeNDImage(NDImage* img, std::string fn, bool nifti2 = true)
{
	std::string mode = "wb";
	const size_t BSIZE = 1024*1024; //1M
	gzFile gz;

	// remove .gz to find the "real" format, 
	std::string fn_nz;
	if(fn.substr(fn.size()-3, 3) == ".gz") {
		fn_nz = fn.substr(0, fn.size()-3);
	} else {
		// if no .gz, then make encoding "transparent"
		fn_nz = fn;
		mode += 'T';
	}
	
	// go ahead and open
	gz = gzopen(fn.c_str(), mode.c_str());
	gzbuffer(gz, BSIZE);

	if(fn_nz.substr(fn_nz.size()-4, 4) == ".nii") {
		if(nifti2) {
			if(writeNifti2Image(img, gz) != 0) {
				std::cerr << "Error writing" << std:: endl;
				gzclose(gz);
				return -1;
			}
		} else {
			if(writeNifti1Image(img, gz) != 0) {
				std::cerr << "Error writing" << std:: endl;
				gzclose(gz);
				return -1;
			}
		}
	} else {
		std::cerr << "Unknown filetype: " << fn_nz.substr(fn_nz.rfind('.')) 
			<< std::endl;
		gzclose(gz);
		return -1;
	}

	gzclose(gz);
	return 0;
}

template <typename T>
NDImage* readNifti1Pixels(gzFile file, nifti1_header* header, bool doswap)
{
	// jump to voxel offset
	gzseek(file, header->vox_offset, SEEK_SET);

	/* 
	 * Create Slicer Object to iterate through image slices
	 */

	// dim 0 is the fastest in nifti images, so go in that order
	std::list<size_t> order(header->ndim, 0);
	std::vector<size_t> dim(header->ndim, 0);
	for(int64_t ii=0; ii<(int64_t)header->ndim; ii++) {
		dim[ii] = header->dim[ii];
		order.push_back(ii);
	}
	
	// TODO HERE
	Slicer slicer(dim, order);

	T tmp;
	NDImage* out;

	// someday this all might be simplify by using NDImage* and the 
	// dbl or int64 functions, as long as we trust that the type is
	// going to be good enough to caputre the underlying pixle type
	switch(header->ndim) {
		case 1: {
			auto typed = new NDImageStore<1, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 2:{
			auto typed = new NDImageStore<2, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 3:{
			auto typed = new NDImageStore<2, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 4:{
			auto typed = new NDImageStore<4, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 5:{
			auto typed = new NDImageStore<5, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 6:{
			auto typed = new NDImageStore<6, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 7:{
			auto typed = new NDImageStore<7, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
				if(doswap) swap<T>(&tmp);
				(*typed)[*slicer] = tmp;
			}
			out = typed;
			} break;
		case 8:{
			auto typed = new NDImageStore<8, T>(dim);
			for(slicer.gotoBegin(); slicer.isEnd(); ++slicer) {
				gzread(file, &tmp, sizeof(T));
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
NDImage* readNifti1Image(gzFile file, bool verbose)
{
	bool doswap = false;
	nifti1_header header;
	if(readNifti1Header(file, &header, &doswap, verbose) < 0)
		return NULL;

	NDImage* out;

	// create image
	switch(header.datatype) {
		// 8 bit
		case NIFTI_TYPE_INT8:
			out = readNifti1Pixels<int8_t>(file, &header, doswap);
		break;
		case NIFTI_TYPE_UINT8:
			out = readNifti1Pixels<uint8_t>(file, &header, doswap);
		break;
		// 16  bit
		case NIFTI_TYPE_INT16:
			out = readNifti1Pixels<int16_t>(file, &header, doswap);
		break;
		case NIFTI_TYPE_UINT16:
			out = readNifti1Pixels<uint16_t>(file, &header, doswap);
		break;
		// 32 bit
		case NIFTI_TYPE_INT32:
			out = readNifti1Pixels<int32_t>(file, &header, doswap);
		break;
		case NIFTI_TYPE_UINT32:
			out = readNifti1Pixels<uint32_t>(file, &header, doswap);
		break;
		case NIFTI_TYPE_FLOAT32:
			out = readNifti1Pixels<float>(file, &header, doswap);
		break;
		// 64 bit
		case NIFTI_TYPE_FLOAT64:
			out = readNifti1Pixels<double>(file, &header, doswap);
		break;
		case NIFTI_TYPE_INT64:
			out = readNifti1Pixels<int64_t>(file, &header, doswap);
		break;
		case NIFTI_TYPE_UINT64:
			out = readNifti1Pixels<uint64_t>(file, &header, doswap);
		break;
		// 128 bit
		case NIFTI_TYPE_COMPLEX64:
			std::cerr << "Single-Precision Complex Images are not currently "
				"supported!" << std::endl;
			return NULL;
		case NIFTI_TYPE_FLOAT128:
			std::cerr << "Quad-Precision Images are not currently "
				"supported!" << std::endl;
			return NULL;
		case NIFTI_TYPE_RGB24:
			std::cerr << "RGB Images are not currently supported!" 
				<< std::endl;
			return NULL;
		break;
		case NIFTI_TYPE_COMPLEX128:
			std::cerr << "Double-Precision Complex not currently supported!" 
				<< std::endl;
			return NULL;
		case NIFTI_TYPE_COMPLEX256:
			std::cerr << "Quad-Precision Complex not currently supported!" 
				<< std::endl;
			return NULL;
		case NIFTI_TYPE_RGBA32:
			std::cerr << "RGBA not currently supported!" << std::endl;
			return NULL;
	}

	if(!out)
		return NULL;

	/* 
	 * Now that we have an Image*, wecan fill in the remaining values from 
	 * the header
	 */

	// figure out orientation
	if(header.qform_code > 0) {
		/*
		 * set spacing 
		 */
		for(size_t ii=0; ii<out->ndim(); ii++)
			out->space(ii) = header.pixdim[ii];
		
		/* 
		 * set origin 
		 */
		// x,y,z
		for(size_t ii=0;ii<3 && ii<out->ndim(); ii++)
			out->origin(ii) = header.qoffset[ii];
		// set toffset
		if(out->ndim() > 3)
			out->origin(3) = header.toffset;
		// set remaining to 0
		for(size_t ii=4;ii<out->ndim(); ii++)
			out->origin(ii) = 0;

		/* Copy Quaternions, and Make Rotation Matrix */

		// copy quaternions, (we'll set a from R)
		out->use_quaterns = true;
		for(size_t ii=0; ii<3 && ii<out->ndim(); ii++)
			out->quaterns[ii+1] = header.quatern[ii];
		
		// calculate a, copy others
		double b = header.quatern[0];
		double c = header.quatern[1];
		double d = header.quatern[2];
		double a = sqrt(1.0-(a*b+c*c+d*d));

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
		if(header.qfac == -1 && out->ndim() > 2)
			out->space(2) = -out->space(2);
		out->updateAffine();
		if(header.qfac == -1 && out->ndim() > 2)
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
		for(int64_t ii=0; ii<header.ndim; ii++)
			out->space(ii) = header.pixdim[ii];
		out->updateAffine();
	}

	/************************************************************************** 
	 * Medical Imaging Varaibles Variables 
	 **************************************************************************/
	
	// direct copies
	out->m_slice_duration = header.slice_duration;
	out->m_slice_start = header.slice_start;
	out->m_slice_end = header.slice_end;
	out->m_freqdim = (int)(header.dim_info.bits.freqdim)-1;
	out->m_phasedim = (int)(header.dim_info.bits.phasedim)-1;
	out->m_slicedim = (int)(header.dim_info.bits.slicedim)-1;

	if(out->m_slicedim > 0) {
		out->m_slice_timing.resize(out->dim(out->m_slicedim), NAN);
	}

	// slice timing
	switch(header.slice_code) {
		case NIFTI_SLICE_SEQ_INC:
			out->m_slice_order = "SEQ";
			for(int ii=out->m_slice_start; ii<=out->m_slice_end; ii++)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_SEQ_DEC:
			out->m_slice_order = "RSEQ";
			for(int ii=out->m_slice_end; ii>=out->m_slice_start; ii--)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_INC:
			out->m_slice_order = "ALT";
			for(int ii=out->m_slice_start; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_start+1; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_DEC:
			out->m_slice_order = "RALT";
			for(int ii=out->m_slice_end; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_end-1; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_INC2:
			out->m_slice_order = "ALT_P1";
			for(int ii=out->m_slice_start+1; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_start; ii<=out->m_slice_end; ii+=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
		case NIFTI_SLICE_ALT_DEC2:
			out->m_slice_order = "RALT_P1";
			for(int ii=out->m_slice_end-1; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
			for(int ii=out->m_slice_end; ii>=out->m_slice_start; ii-=2)
				out->m_slice_timing[ii] = ii*out->m_slice_duration;
		break;
	}

	return out;
}

//NDImage* readNifti2Image(gzFile file)
//{
//	nifti2_header header;
//	static_assert(sizeof(header) == 540, "Error, nifti header packing failed");
//
//	gzread(file, &header, sizeof(header));
//	std::cerr << header.magic << std::endl;
//	if(strcmp(header.magic, "n+2")) {
//		gzclearerr(file);
//		gzrewind(file);
//		return NULL;
//	}
//
//	NDImage* out;
//	// valid, go ahead and parse
//	
//	return out;
//}
//
int writeNifti1Image(NDImage* out, gzFile file)
{
	const size_t HEADERSIZE = 348;
	nifti1_header header;
	static_assert(sizeof(nifti1_header) == HEADERSIZE, "Error, nifti header packing failed");

	std::fill((char*)&header, ((char*)&header)+HEADERSIZE, 0);

	header.sizeof_hdr = HEADERSIZE;

	if(out->m_freqdim > 0)  header.dim_info.bits.freqdim = out->m_freqdim+1;
	if(out->m_phasedim > 0)  header.dim_info.bits.phasedim = out->m_phasedim+1;
	if(out->m_slicedim > 0)  header.dim_info.bits.slicedim = out->m_slicedim+1;

	header.ndim = out->ndim();
	for(size_t dd=0; dd<out->ndim(); dd++) {
		header.dim[dd] = out->dim(dd);
		header.pixdim[dd] = out->space(dd);
	}

	std::cerr << "Error NiftiWriter not yet implemented" << std::endl;
	throw (-1);
	
//	if(out->quatern) {
//		
//	}
//	double a = 0.5*sqrt(1+R11+R22+R33);
//    header.quatern_b = 0.25*(R32-R23)/a;
//	header.quatern_c = 0.25*(R13-R31)/a;
//	header.quatern_d = 0.25*(R21-R12)/a
//
//	// read the pixels
//	// note x is the fastest in nifti, for us it is the slowest
//	switch(header.datatype) {
//		case DT_INT32:
//			break;
//		case DT_FLOAT:
//			break;
//	}
	return 0;
}

int writeNifti2Image(NDImage* out, gzFile file)
{
	std::cerr << "Error NiftiWriter not yet implemented" << std::endl;
	throw (-1);
	nifti2_header header;
	static_assert(sizeof(header) == 540, "Error, nifti header packing failed");

	return 0;
}

