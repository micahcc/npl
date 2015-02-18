#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.1.1-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.1.1-release"
	#else
		#define __version__ "3.1.1"
	#endif
#endif
