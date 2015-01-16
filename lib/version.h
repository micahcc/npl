#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.0.6-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.0.6-release"
	#else
		#define __version__ "3.0.6"
	#endif
#endif
