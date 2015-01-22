#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.0.8-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.0.8-release"
	#else
		#define __version__ "3.0.8"
	#endif
#endif
