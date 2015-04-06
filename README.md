NPL: Neuro Programs and Libraries
=======

Building
=======
To build download and cd into the root directory. Then run the following commands:

$ git submodule init

$ git submodule update

$ ./waf configure --prefix=INSTALLDIR --release

$ ./waf install -j 4

Once you install you will need to add INSTALLDIR/lib to your LD_LIBRARY_PATH. Alternatively you can use --enable-rpath on the configure line to get binaries to look in ../lib for the necessary libraries. 

Tools
=======

Motion Correction

Linear Regression of Entire fMRI vs. Input Matrix

fMRI Distortion Correction

Tools (WIP)
==========
fMRI ICA

fMRI Group ICA

Need to
========
Create Website with doxygen output

Add some simple examples
