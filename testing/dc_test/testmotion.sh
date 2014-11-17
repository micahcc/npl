#!/bin/bash
set -x
baseimg=orig.nii.gz
distimg=field.nii.gz

# create motion file
echo 20 20 20 0 0 0 0 0 0 > motion0.rtm
echo 20 20 20 .1 .2 -0.3 3 2 4 > motion1.rtm

# apply distortion to 0
nplConvertDeform -i $distimg --uni-space +y -o dist_0iv.nii.gz --invert
nplConvertDeform -i $distimg --uni-space +y -o dist_0v.nii.gz 
nplApplyDeform -i $baseimg -d dist_0iv.nii.gz -o sim_0d.nii.gz
nplApplyDeform -i sim_0d.nii.gz -d dist_0v.nii.gz -o sim_0dc.nii.gz

# apply distortion to 1, after rotating
nplMotionCorr --invert -a motion1.rtm -i $baseimg -o sim_1m.nii.gz
nplMotionCorr --invert -a motion1.rtm -i $distimg -o dist_1m.nii.gz
nplConvertDeform -i dist_1m.nii.gz --uni-space +y -o dist_1miv.nii.gz --invert
nplApplyDeform -i sim_1m.nii.gz -d dist_1miv.nii.gz -o sim_1md.nii.gz

# test the inverse
nplConvertDeform -i dist_1m.nii.gz --uni-space +y -o dist_1mv.nii.gz 
nplApplyDeform -i sim_1md.nii.gz -d dist_1mv.nii.gz -o sim_1mdc.nii.gz
nplMotionCorr -a motion1.rtm -i sim_1mdc.nii.gz -o sim_1mdcc.nii.gz

# Create Brute Force Corrected
fslmerge -t sim_mdcc.nii.gz sim_0dc.nii.gz sim_1mdcc.nii.gz 

# Joint to Create Distorted
fslmerge -t sim.nii.gz sim_0d.nii.gz sim_1md.nii.gz 
cat motion0.rtm motion1.rtm > motion.rtm

# test, sim_dc.nii.gz should match baseimg
nplMotionCorr -a motion.rtm -i sim.nii.gz -o sim_mc.nii.gz 
nplDistortionCorr -R motion.rtm -a $distimg -d y -m sim.nii.gz -o sim_dc.nii.gz

comp=`nplCompare sim_dc.nii.gz sim_mdcc.nii.gz -m cor`
if (( $(echo "$comp < 0.995" | bc -l) )); then
	echo "ERROR DIFFERENCES DETECTED!" 
	exit -1
fi
