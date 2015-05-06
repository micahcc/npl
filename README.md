# NPL: Neuro Programs and Libraries

## Building
To build download and cd into the root directory. Then run the following commands:

    $ git submodule init
    $ git submodule update
    $ ./waf configure --prefix=INSTALLDIR --release
    $ ./waf install -j 4

Once you install you will need to add INSTALLDIR/lib to your LD_LIBRARY_PATH. Alternatively you can use --enable-rpath on the configure line to get binaries to look in ../lib for the necessary libraries.

## Tools

### Motion Correction
Motion correction is pretty simple, it computes rigid transforms to the 0th timepoint using correlation. The command line is similarly easy:

        $ nplMotionCorr -i fmrni.niigz -o fmri_mc.nii.gz -a motion.txt

### Bias Field Correction
Bias field correction uses a brain mask, or estimates a brain mask (internally, using OTSU thresholding), then computes a B-spline which fits the low frequency intensity shifts across the masked part of an input image. An example usage may be:

        $ nplBiasFieldCorrect -i T1.nii.gz -c T1_bc.nii.gz -b T1_bf.nii.gz

Where `T1_bc.nii.gz` is the bias-corrected output and `T1_bf.nii.gz` is the resulting bias field. Additional arguments are availabe which set the knot spacing (`-s`), tikhonov regularization weight (`-R`) and mask (`-m`).

### Cleaning fMRI of Structured Noise
It is necessary to remove structured noise from fMRI before analysis, especially when correlating signals across the brian. `nplCleanFMRI` is a program to perform cleaning of fMRI data prior to other analysis. An exmaple:

        $ nplCleanFMRI -i fmri.nii.gz -o fmri_clean.nii.gz -f 0.00001 -F 0.01 -r motion.txt -P -c 7 -L labelmap.nii.gz -l 0,1,2,3

This extracts labels 0-3 and takes the 7 leading principal components (because of the argument `-P -c 7`) as additional confounds to be regressed out. It then regresses out the motion parameters provided in a 6 column time-series csv file (`motion.txt`), and the leading principal components. Finally frequencies 0.00001Hz - 0.01Hz are kept, and everything outside is filtered out. The final output is called `fmri_clean.nii.gz`

### fMRI Distortion Correction
Correcting distortion using registration may be done using a T1 image that has been rigidly registered to the fMRI.  Example usage:

        $ nplDistortionCorr --moving fmri.nii.gz --fixed t1.nii.gz -S 3 -S 2 -S 1 --direction x --out fmri_undist.nii.gz

This sets the smoothing schedule to 3mm, 2mm, 1mm. The default is finer grain but this shows how the schedule can be modified: with repeated calls to -S. The direction of phase-encoding is EXTREMELY important. This is the direction of distortion. Additional parameters can change the optimization process and distortion field. In particular `--bspline-space` sets the knot spacing: smaller values will allow for larger distortion. See --help for more.

### Rigid Registration
Rigid registration can be carried out either using Mutual Information or Correlation. Example:

        $ nplRigidReg -m source.nii.gz -f target.nii.gz -M MI -o src_in_tgt_space.nii.gz -s 5 -s 4 -s 3 -s 2 -s 1 -s 0

Like distortion correction, it is possible to customize the smoothing steps with multiple calls to `-s`. The metric may be either mutual information (`MI`, for multi-modal) or correlation (`COR`, for intra-modal). By default the resulting image (`src_in_tgt_space.nii.gz`) is not re-sampled, but just has its orientation set to reflect the rotation and translation. To maintain the original orientation add the `--resample` option.

### fMRI Single Subject ICA
Single subject ICA is pretty easy, and can be performed completely in memory. The command line is short:

        $ nplICA -i fmri.nii.gz -v 0.9 -t tmap.nii.gz

Where 90% of the input variance is maintained (`-v`).

### fMRI Group ICA
Group ICA is actually performed by first reducing the data into chunks, then performing out-of-memory PCA, followed by ICA. Thus it actually takes 3 executables to perform these steps:

        $ nplGICA_reorg --reorg-prefix $orgpref -v -M 2 -m mask.nii.gz -i in1.nii.gz -i in2.nii.gz ...
        $ nplGICA_reduce --var-thresh 0.2 --power-iters 3 --rank 100 --reorg-prefix $orgpref --reduce-prefix $repref
        $ nplGICA_ica --reorg-prefix $orgpref --reduce-prefix $repref --ica-prefix $icapref -v

There is a good deal going on here. First reorganizing the data into matrices is necessary to limit memory usage. The size of each chunk (in gigabytes) is set by `-M` for `nplGICA_reorg`, and memory should not go above that any of the further processing steps. The prefixes set the location of the outputs from each step; note that all the prefixes can actually be the same. Images can be both spatially and temporally concatenated -- for instance if there are multiple runs of a single subject, where subjects were each performing a known task. Space should be considered as columns and time as rows. Data is entered as a flat array on the command line for `nplGICA_reorg` using multiple `-i` flags, each with different image. Thus multiple fMRI images can be concatenated spatially and temporally as shown:

<table>
<tr><td></td><td>Space 0</td><td>Space 1</td></tr>
<tr><td>Time 0</td><td>-i 0</td><td>-i 1</td></tr>
<tr><td>Time 1</td><td>-i 2</td><td>-i 3</td></tr>
<tr><td>Time 2</td><td>-i 4</td><td>-i 5</td></tr>
<tr><td>Time 3</td><td>-i 6</td><td>-i 7</td></tr>
<tr><td>Time 4</td><td>-i 8</td><td>-i 9</td></tr>
</table>

This would correspond to the command line:

        $ nplGICA_reorg $orgpref -i 0 -i 1 -i 2 -i 3 -i 4 -i 5 -i 6 -i 7 -i 8 -i 9 -m m0 -m m1

Note that if no masks are given, then images 0 and 1 will be thresholded and the resulting masks will be used in place `m0` and `m1`. Thus masks correspond to columns not individual images. This makes sense, given that temporally concatenated data should be aligned.

PCA data reduction takes the most parameters, although generally the ones given above should work. `--var-thresh` determines the ratio of the maximal eigenvalue to stop at, when performing a low rank approximation. `--rank` sets the maximum rank to use in the same approximation. `--power-iters` basically costs computation but improves the randomized subspace approximation.

ICA itself does not have many options, one not shown above is `-T` which causes the program to compute a temporal ICA rather than spatial.

### Image Math
An image math tool is provided to be able to quickly compute math operations on images. The advantage over fslmaths is that it can obey orientation, so images with different spacing will be automatically re-sampled into the space of the first image passed on the command line. Example usage:

        $ nplMath -c imgC.nii.gz -a imgA.nii.gz -b imgB.nii.gz 'a*b-c' --out result.nii.gz

There are several important points here. First: the `result.nii.gz` image will have the same spacing and orientation as `imgC.nii.gz`. Its worth noting that `'a*b'` would still be a valid equation and the output would still be in the space of imgC.nii.gz but that would not be used for any math.  Interpolation is done only if the images have different orientation. Valid operations include: logical operators (`|`, `&`, `neg`), logical comparisons (`>`, `>=`, `==`, `<=`, `<`), basic arithmetic (`*`, `+`, `-`, `/`), trig functions (`cos`, `sin`, `tan`), and a few others (`ceil`, `floor`, `exp`, `round`, `log`).

## API
The API can be used to load images and do basic operations on image, much as you can do in other ND-Image API's. The Doxygen output currently resides [at UCLA brainmapping](http://users.bmap.ucla.edu/~mchamber/npl/index.html).

### TODO simple examples with the API
