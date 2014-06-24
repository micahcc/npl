
#include "ndimage.h"
#include "nifti.h"

#include "zlib.h"

#include <cstring>

/* Functions */
NDImage* readNDImage(std::string filename);
void writeNDImage(NDImage* img, std::string filename);
NDImage* readNifti1Image(gzFile file);
NDImage* readNifti2Image(gzFile file);
int writeNifti1Image(NDImage* out, gzFile file);
int writeNifti2Image(NDImage* out, gzFile file);

/* Pre-Compile Certain Image Types */
class NDImageStore<1, float>;
class NDImageStore<1, double>;
class NDImageStore<1, int>;

class NDImageStore<2, float>;
class NDImageStore<2, double>;
class NDImageStore<2, int>;

class NDImageStore<3, float>;
class NDImageStore<3, double>;
class NDImageStore<3, int>;

class NDImageStore<4, float>;
class NDImageStore<4, double>;
class NDImageStore<4, int>;

class NDImageStore<5, float>;
class NDImageStore<5, double>;
class NDImageStore<5, int>;

class NDImageStore<6, float>;
class NDImageStore<6, double>;
class NDImageStore<6, int>;

class NDImageStore<7, float>;
class NDImageStore<7, double>;
class NDImageStore<7, int>;

class NDImageStore<8, float>;
class NDImageStore<8, double>;
class NDImageStore<8, int>;


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

/* 
 * Nifti Readers 
 */

NDImage* readNifti1Image(gzFile file)
{
	nifti1_header header;
	static_assert(sizeof(header) == 348, "Error, nifti header packing failed");

	gzread(file, &header, sizeof(header));
	std::cerr << header.magic << std::endl;
	if(strcmp(header.magic, "n+1")) {
		gzrewind(file);
		return NULL;
	}

	NDImage* out;
	
	// figure out orientation
	if(header.qform_code == 0) {
		
	} else {

	}

	// valid, go ahead and parse
	return out;
}

NDImage* readNifti2Image(gzFile file)
{
	nifti2_header header;
	static_assert(sizeof(header) == 540, "Error, nifti header packing failed");

	gzread(file, &header, sizeof(header));
	std::cerr << header.magic << std::endl;
	if(strcmp(header.magic, "n+2")) {
		gzclearerr(file);
		gzrewind(file);
		return NULL;
	}

	NDImage* out;
	// valid, go ahead and parse
	
	return out;
}

int writeNifti1Image(NDImage* out, gzFile file)
{
	size_t HEADERSIZE = 348;
	nifti1_header header;
	static_assert(sizeof(header) == HEADERSIZE, "Error, nifti header packing failed");

	std::fill((char*)&header, ((char*)&header)+HEADERSIZE, 0);

	header.sizeof_hdr = HEADERSIZE;

	if(out->freqdir >= 0) header.dim_info.bits.freqdim = out->freqdim+1;
	if(out->phasedir >= 0) header.dim_info.bits.phasedim = out->phasedim+1;
	if(out->slicedir >= 0) header.dim_info.bits.slicedim = out->slicedim+1;

	header.ndim = out->getNDim();
	for(size_t dd=0; dd<out->getNDim(); dd++) {
		header.dim[dd] = out->dim(dd);
		header.pixdim[dd] = out->m_space[dd];
	}
	
	double a = 0.5*sqrt(1+R11+R22+R33);
    header.quatern_b = 0.25*(R32-R23)/a;
	header.quatern_c = 0.25*(R13-R31)/a;
	header.quatern_d = 0.25*(R21-R12)/a

	// read the pixels
	// note x is the fastest in nifti, for us it is the slowest
	switch(header.datatype) {
		case DT_INT32:
			break;
		case DT_FLOAT:
			break;
	}
	return 0;
}

int writeNifti2Image(NDImage* out, gzFile file)
{
	nifti2_header header;
	static_assert(sizeof(header) == 540, "Error, nifti header packing failed");

	return 0;
}
