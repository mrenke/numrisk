# DTI data analsyis


## Zoltans proposal for FA images (+ChatGPT)

1. Topup for field-map correction
2. Create a Brain Mask
3. Prepare Gradient Information
(4. 4. Apply Eddy Current and Motion Correction)

5. Fit the Diffusion Tensor Model

### 1. Topup for field-map correction

* Step 1: DataPrep 
- Merge Your Blip-Up and Blip-Down Data: 
fslmerge -t dwi_as sn_*_as_bvalue2_diffori*.nii
fslmerge -t dwi_ps sn_*_ps_bvalue2_diffori*.nii
fslmerge -t b0_all sn_*_as_bvalue1_diffori33.nii sn_*_ps_bvalue1_diffori33.nii
fslmerge -t dwi_all dwi_as dwi_ps

- Create a text file (acqparams.txt) describing the phase encoding direction and echo time for TOPUP. It usually looks like this:
0 1 0 0.05  # blip-up phase encoding (AP or PA)
0 -1 0 0.05 # blip-down phase encoding (PA or AP)
* Step 2:  Run TOPUP for Susceptibility Correction 
topup --imain=dwi_all --datain=acqparams.txt --config=b02b0.cnf --out=topup_results --iout=corrected_b0
--> seems to work with FSL installation on sciencecloud:
topup --imain=b0_all --datain=acqparams.txt --config=b02b0.cnf --subsamp=1 --out=topup_results --iout=corrected_b0

### 2. Create a Brain Mask

### 3. Prepare Gradient Information
* bvals: (one row of numbers for diffusion strengths)
    nr of b-factors =		2;
    b-factors =			0, 1000, (30) 0;

* bvecs = three rows (x, y, z) of diffusion directions.


### 5. Fit the Diffusion Tensor Model
Run dtifit to calculate FA:
dtifit -k eddy_corrected -o dti -m brain_mask -r bvecs -b bvals




# Tract-Based Spatial Statistics (TBSS)
https://andysbrainbook.readthedocs.io/en/stable/TBSS/TBSS_Overview.html

--> wants all individual 


does not work!! 

❯ tbss_1_preproc sub_2_3_merged_FAs.nii.gz
processing sub_2_3_merged_FAs

Fatal error: cannot open file: -
    0: (main)


Fatal error: cannot open file: -
    0: (main)


Fatal error: cannot open file: -
    0: (main)

/usr/local/fsl/bin/tbss_1_preproc: line 86: 44546 Segmentation fault: 11  $FSLDIR/bin/fslmaths $f -min 1 -ero -roi 1 $X 1 $Y 1 $Z 0 1 FA/${f}_FA
Image Exception : #63 :: No image files match: FA/sub_2_3_merged_FAs_FA
libc++abi: terminating due to uncaught exception of type std::runtime_error: No image files match: FA/sub_2_3_merged_FAs_FA
/usr/local/fsl/bin/tbss_1_preproc: line 86: 44591 Abort trap: 6           $FSLDIR/bin/fslmaths FA/${f}_FA -bin FA/${f}_FA_mask
Image Exception : #63 :: No image files match: FA/sub_2_3_merged_FAs_FA_mask
libc++abi: terminating due to uncaught exception of type std::runtime_error: No image files match: FA/sub_2_3_merged_FAs_FA_mask
/usr/local/fsl/bin/tbss_1_preproc: line 86: 44592 Abort trap: 6           $FSLDIR/bin/fslmaths FA/${f}_FA_mask -dilD -dilD -sub 1 -abs -add FA/${f}_FA_mask FA/${f}_FA_mask -odt char
Now running "slicesdir" to generate report of all input images
-e <thr>   :  use the specified threshold for edges (if >0 use this proportion of max-min, if <0, use the absolute value)
-S         : output every second axial slice rather than just 9 ortho slices




# OLD: with eddy:
7. Eddy:
eddy --imain=dwi_all --mask=nodif_brain_mask \                                              
    --acqp=acqparams.txt --index=index.txt \
    --bvecs=dwi_all.bvecs --bvals=dwi_all.bval \
    --topup=topup_results --out=eddy_corrected
8. dtifit: 
dtifit --data=eddy_corrected \                                                              
       --mask=nodif_brain_mask \
       --bvecs=eddy_corrected.eddy_rotated_bvecs \
       --bvals=dwi_all.bval \
       --out=dtifit_results


### Command from FSL GUI dtifit:
/usr/local/fsl/bin/dtifit --data=/Volumes/mrenkeED/data/ds-dnumrisk/sourcedata/tryout_DTI/sub-02/dwi_all.nii.gz --out=/Volumes/mrenkeED/data/ds-dnumrisk/sourcedata/tryout_DTI/sub-02/dtifit_results_GUI --mask=/Volumes/mrenkeED/data/ds-dnumrisk/sourcedata/tryout_DTI/sub-02/nodif_brain_mask.nii.gz --bvecs=/Volumes/mrenkeED/data/ds-dnumrisk/sourcedata/tryout_DTI/sub-01/bvecs_all_T.txt --bvals=/Volumes/mrenkeED/data/ds-dnumrisk/sourcedata/tryout_DTI/sub-01/bvals_all_T.txt --wls


what worked: 
dtifit --data=dwi_topup_corrected --mask=mask_dwi_topup_corrected_mean --bvecs=dwi_all.bvecs --bvals=dwi_all.bval --out=dtifit_meammask_results