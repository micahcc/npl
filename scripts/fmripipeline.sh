#!/bin/bash
set -x
if [[ $# -ne 3 ]]; then
	echo "$0 T1Brain fmri Working"
	exit -1
fi

#UnwarpDim=`echo $UnwarpDir | tr -d '-'`
UnwarpDim='x'
NPLDIR=/ifs/students/mchambers/npl-3.0/
T1wBrainImageFile=$1
fmri=$2
bias=$3

if [[ -z "$smooth" ]]; then
	smooth="-S 3 -S 2.5 -S 2 -S 1.5 -S 1 -S 0.5 -S 0.1 -S 0"
fi

if [[ -z "$jac" ]]; then
	jac="0.15"
fi

if [[ -z "$tps" ]]; then
	tps="0"
fi

if [[ -z "$bins" ]]; then
	bins=150
fi

if [[ -z "$brad" ]]; then
	brad=4
fi

if [[ -z "$knotdist" ]]; then
	knotdist=16
fi

echo "Smoothing: $smooth"
echo "Jacobian Reg Weight: $jac"
echo "Thin-Plate Spline Weight: $tps"
echo "Number of Bins: $bins"
echo "Bin Radius: $brad"

# Perform Motion Correction (or if motion correction is done, then just apply)
if [[ -e fmri_motion.txt ]]; then
	${NPLDIR}/bin/nplMotionCorr -a fmri_motion.txt -o fmri_mc.nii.gz -i $fmri
else
	${NPLDIR}/bin/nplMotionCorr -m fmri_motion.txt -o fmri_mc.nii.gz -i $fmri
fi

# Average Image to get better SNR (or comment out and just use fslroi)
${NPLDIR}/bin/nplReduce4D -i fmri_mc.nii.gz -m fmri_avg.nii.gz

echo Move BIAS image to fMRI Space
${NPLDIR}/bin/nplBiasFieldCorrect -i $bias -b biasfield.nii.gz -R .1 -c bias_bc.nii.gz
${NPLDIR}/bin/nplGuessOrient -m bias_bc.nii.gz -f fmri_avg.nii.gz -o bias_init.nii.gz
${NPLDIR}/bin/nplRigidReg -m bias_init.nii.gz -f fmri_avg.nii.gz -M MI -o bias_fspace.nii.gz -s 5 -s 4 -s 3 -s 2 -s 1 -s 0
${NPLDIR}/bin/nplOrientToTransform -t bf_fspace.rtm -m $bias -f bias_fspace.nii.gz
${NPLDIR}/bin/nplRigidReg -m biasfield.nii.gz --apply bf_fspace.rtm -o biasfield_fspace.nii.gz

echo Estimate Bias field
${NPLDIR}/bin/nplMath --lin --double -b fmri_avg.nii.gz -a biasfield_fspace.nii.gz --out fmri_bc.nii.gz "b/a"

echo Move Bias T1w Image to fMRI Space
${NPLDIR}/bin/nplGuessOrient -m ${T1wBrainImageFile}  -f fmri_bc.nii.gz -o brain_init.nii.gz
${NPLDIR}/bin/nplRigidReg -m brain_init.nii.gz -f fmri_bc.nii.gz -M MI -o brain_fspace.nii.gz

## Skull Strip
bet fmri_bc.nii.gz fmri_brain.nii.gz -F

# rigid fmri to reference T1
time ${NPLDIR}/bin/nplDistortionCorr --moving fmri_brain.nii.gz \
--fixed brain_fspace.nii.gz --otsu-thresh $smooth \
--direction $UnwarpDim  --metric MI --jacreg $jac --tpsreg $tps \
--transform npl_fmri_dist.nii.gz \
-F npl_field.nii.gz --out fmri_undist.nii.gz --bins $bins --radius $brad \
--bspline-space $knotdist
