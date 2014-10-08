#!/bin/bash

NPLDIR=/ifs/students/mchambers/npl-3.0/

######
# This is Supposed to be a pipeline that will do fmri processing. 
# It needs updating.
# - Base around an output directory, and don't recompute unless -f is passed
# - Input a bias-field estimation image (or if none is given use T1)
# - Change bias field estimation to my algorithm
# - Change bias field correction to nplMath
# - Change rigid registration my rigid registration
#    - Create format for storing rigid transform
# - Change Motion Correction to my motion correction
#    - Make averaging an option

usage()
{
	echo "$0 -o outdir -r reference -f fmri -m method -l labelmap"
	echo "This program takes a reference image (usually t1), an fMRI "
	echo "image, and an output dir. Optionally you can set the method of"
	echo "dealing with motion. Options are: dall, d1, rigid. dall indicates"
	echo "that all timepoints should be distortion corrected, d1 indicates "
	echo "only timepoint one should be distortion-corrected (all the "
	echo "others will use a similar distortion field. rigid causes "
	echo "no distortion correction to be performed, instead just rigid"
	echo "motion correction will be done."
	exit $1
}

bfc=""
t1=""
fmri=""
outdir=""
while getopts o:f:r:m:R:a:l:h opt; do
	case $opt in
		o)
			outdir=$OPTARG
		;;
		f)
			fmri=$OPTARG
		;;
		r)
			t1=$OPTARG
		;;
		m)
			method=$OPTARG
		;;
		h)
			usage(0)
		;;
		\?)
			echo "Invalid option: -$OPTARG" >&2 
			usage(-1)
		;;
	esac
done

if [ -z "$bfc" ]; then
	bfc=$t1;
fi

echo "Input FMRI: " $fmri
echo "Input T1: " $t1
echo "Output Directory: " $outdir

if [ ! -e "$t1" ] || [ ! -e "$fmri" ] || [ ! -e "$bfc" ] || [ ! -d "$outdir" ];
then
	echo "Error in input"
	usage(-1)
fi

for i in `seq 5`; do
	echo $i
	sleep 1
done

if [[ "$upscale" == "" ]]; then 
	upscale=1
	echo 'setting upscale=' $upscale
else
	echo 'using upscale =' $upscale
fi
	
if [[ "$rescale" == "" ]]; then 
	rescale=1
	echo 'setting rescale =' $rescale
else
	echo 'using rescale =' $rescale
fi

if [[ "$cost" == "" ]]; then 
	cost="-C MI -C MI -C MI -C VI -C VI"
	echo 'setting cost =' $cost
else
	echo 'using cost =' $cost
fi

if [[ "$smooth" == "" ]]; then 
	smooth="-S 2 -S 1.5 -S 1 -S 0.5 -S 0"
	echo 'setting smooth =' $smooth
else
	echo 'using smooth =' $smooth
fi

if [[ "$optimizer" == "" ]]; then 
	optimizer="-O LBFGS -O grad -O LBFGS -O grad -O LBFGS"
	echo 'setting optimizer =' $optimizer
else
	echo 'using optizer =' $optimizer
fi

if [[ "$scale" == "" ]]; then 
	scale=1000
else
	echo 'using scale =' $scale
fi

if [[ "$gstop" == "" ]]; then 
	gstop=0.001
else
	echo 'using gstop =' $gstop
fi

if [[ "$fstop" == "" ]]; then 
	fstop=0.001
	echo 'setting fstop =' $fstop
else
	echo 'using fstop =' $fstop
fi

if [[ "$alpha_tps" == "" ]]; then 
	alpha_tps=0.001
	echo 'setting alpha_tps =' $alpha_tps
else
	echo 'using alpha_tps =' $alpha_tps
fi

if [[ "$alpha_jac" == "" ]]; then 
	alpha_jac=0.0001
	echo 'setting alpha_jac =' $alpha_jac
else
	echo 'using alpha_jac =' $alpha_jac
fi

if [[ "$hist" == "" ]]; then 
	hist=4
	echo 'setting hist =' $hist
else
	echo 'using hist =' $hist
fi

if [[ "$dist" == "" ]]; then 
	dist=12
	echo 'setting dist =' $dist
else
	echo 'using dist =' $dist
fi

if [[ "$bins" == "" ]]; then 
	bins=128
	echo 'setting bins =' $bins
else
	echo 'using bins =' $bins
fi

if [[ "$kern" == "" ]]; then 
	kern=4
	echo 'setting kern =' $kern
else
	echo 'using kern =' $kern
fi

set -e
set -x

mkdir -p $outdir

fmriclean()
{
	fmri=$1
	motion=$2
	clean=$3
	label=$4
	# 0 bg
	# 1,6 CSF 
	# 2,4,5 GM
	# 3 WM
	if [ -e $label ]; then
		$NPLDIR/bin/fIRegress -i $fmri -H cannonical -d -r $motion -P -L $label\
		-o ${clean%.nii.gz}_regress.nii.gz -l 0,1,6,3
	else 
		$NPLDIR/bin/fIRegress -i $fmri -H cannonical -d -r $motion \
		-o ${clean%.nii.gz}_regress.nii.gz
	fi
	$NPLDIR/bin/fITimeFilter -i ${clean%.nii.gz}_regress.nii.gz \
		-H .0083 -L .15 -o $clean
}

biasfieldcorrect() {
	orig=$1
	tlabel=$2
	tbiascor=$3

	lbase=${tlabel%.nii.gz}
	bcbase=${tbiascor%.nii.gz}

	# for really high bias fields, first do a low threshold, then a stricter one
	# fmri -> VERY basic label
	# pthresh breaks stuff, especially near image boundaries
#	$NPLDIR/bin/imgSimpleSegment -m pthresh -i $orig -o ${lbase}_1.nii.gz
	$NPLDIR/bin/imgSimpleSegment -m expmax -i $orig -c 3 -o ${lbase}_1.nii.gz
	
	# fmri -> VERY basic bias correction
	$NPLDIR/bin/imgSimpleBiasCorrect -i $orig -L ${lbase}_1.nii.gz -o ${bcbase}_1.nii.gz
	cp -v ${bcbase}_1.nii.gz $tbiascor
	
#	for i in `seq 2 3`; do
#		# VERY basic bias corrected fmri -> EM labels
##		$NPLDIR/bin/imgSimpleSegment -m expmax -i ${bcbase}_$((i-1)).nii.gz -c $i -o ${lbase}_$i.nii.gz
#		$NPLDIR/bin/imgSimpleSegment -m expmax -i ${bcbase}_$((i-1)).nii.gz -c 3 -o ${lbase}_$i.nii.gz
#
#		# fmri, EM labels -> fmri bias field corrected
#		$NPLDIR/bin/imgSimpleBiasCorrect -i $orig -L ${lbase}_$i.nii.gz -o ${bcbase}_$i.nii.gz
#		cp -v ${bcbase}_$i.nii.gz $tbiascor
#	done

}

rigidreg()
{
	fixed=$1
	moving=$2
	outtfm=$3
	outimg=$4
	/ifs/students/mchambers/brainsfit/BRAINSFit --initializeTransformMode \
			useMomentsAlign --useRigid --movingVolume $moving  \
			--fixedVolume $fixed --outputVolume ${outimg%.nii.gz}_low.nii.gz \
			--linearTransform $outtfm

	 /ifs/students/mchambers/npl-latest/bin/imgApplyTfm -t $outtfm -i $moving \
	 		-o $outimg -r $fixed -k
}

distortioncorrect()
{
	in=$1
	ref=$2
	fieldmap=$3
	mc=$4
	mcparams=$5
	out=$6
	justone=$7

	$NPLDIR/bin/fIDistortionCorrect --input $in \
		--t1 $ref --out $out --field-map $fieldmap $justone --debug-level 4 \
		--knot-dist $dist --mc-only $mc --average ${mc%.nii.gz}_avg.nii.gz \
		--motion $mcparams --bd-scale $scale \
	 	--TPSReg $alpha_tps --JacReg $alpha_jac \
		--bd-iters 10000 --bd-lbfgs-hist $hist \
		--bd-grad-rescale $rescale \
	 	$cost $optimizer $smooth --bd-bins $bins \
		--bd-kern-rad $kern --upscale-avg $upscale \
	 	--bd-gstop $gstop --bd-fstop $fstop 
}

# bias field correct
biasfieldcorrect $rawfmri $outdir/fmri_bclabel.nii.gz \
	$outdir/fmri_bc.nii.gz 

biasfieldcorrect $rawt1 $outdir/t1_bclabel.nii.gz $outdir/t1_bc.nii.gz 

# T1 -> fmri
rigidreg $outdir/fmri_bc.nii.gz $outdir/t1_bc.nii.gz $outdir/t1_to_fsp.tfm \
	$outdir/t1_bc_fsp.nii.gz 

$NPLDIR/bin/imgROI -i $outdir/fmri_bc.nii.gz -o $outdir/fmriref.nii.gz -t 0 -t 1

/ifs/students/mchambers/brainsfit/BRAINSResample --inputVolume $labelmap \
	--warpTransform $outdir/t1_to_fsp.tfm --interpolationMode NearestNeighbor \
	--outputVolume $oudir/tissuelabels.nii.gz --pixelType short \
	--referenceVolume fmriref.nii.gz 

if [ "$method" == "d1" ]; then
	# T1 -> fmri
	distortioncorrect $outdir/fmri_bc.nii.gz $outdir/t1_bc_fsp.nii.gz \
		$outdir/fieldmap.nii.gz $outdir/mc.nii.gz $outdir/motion.txt \
		$outdir/dc.nii.gz -1

elif [ "$method" == "dall" ]; then
	distortioncorrect $outdir/fmri_bc.nii.gz $outdir/t1_bc_fsp.nii.gz \
		$outdir/fieldmap.nii.gz $outdir/mc.nii.gz $outdir/motion.txt \
		$outdir/dc.nii.gz ""
else
	$NPLDIR/bin/fIBasicMotionCorrect -i $outdir/fmri_bc.nii.gz \
				-o $outdir/mc.nii.gz -M $outdir/motion.txt -O
	cp $outdir/mc.nii.gz $outdir/dc.nii.gz
fi

fmriclean $outdir/dc.nii.gz $outdir/motion.txt $outdir/fmri_pp.nii.gz $labelmap
