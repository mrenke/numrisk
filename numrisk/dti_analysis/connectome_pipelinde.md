# Trying the connectome pipeline from Anna Speckert

https://github.com/annspe/connectome_pipeline?tab=readme-ov-file

### set up new sciencecloud instance (sciencecloudC): 
- environment connectome_pipeline -- requirements without version specifications
- install ANTs, FSL, MTRix3 & Freesurfer
- copy some freesurfer necessary file into /opt/freesurfer/freesufer<version>/....Color.txt (dont know where this is hard coded and why it is not taking my real freesurfer_dir.. though, when I set it up (source setupFreesurfer.sh)) - whole environment does not work anymore (which python = ...fsl)
- make ANTS commands executabel: export PATH=/home/ubuntu/git/ants.../bin:$PATH (`export PATH=/home/ubuntu/git/ants-2.6.0/bin:$PATH`)
--> problem: for source 

#### Changes/Adaptations

- csd_total_CS5.sh,line 29, biascorr.nii.gz --> biascorr.mif
- had to copy bvecs and bvals into "sub-04/processing/dti" folder manually ("dwi2fod: [ERROR] input file "../../dti/bvecs" for option "-fslgrad" not found")

- last step of DTI_pipeline.py : had to change `three_tissue` into `two_tissue` ({"rfe": "SS2T"}, not "SS3T"... )

## Prep Data:
copy on sciencecloud from /mnt_03/ds-dnumrisk into /mnt_AdaBD....

create t2 & dti folder in sub-folders

- ... <base_dir>/<subj_id>/t2:
            T2_SVRTK.nii.gz # a good anatomical image, not necessarily t2 - from fmriprep/sub-XX/ses-1/anat/sub-01_ses-1_desc-preproc_T1w.nii.gz
            aparc+aseg.mgz # from freesurfer/sub-XX/mri/
cp /mnt_03/ds-dnumrisk/derivatives/freesurfer/sub-01/mri/aparc+aseg.mgz /mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/dwi_forAnnasCP/sub-01/t2/aparc+aseg.mgz # can also rename!
loop to run on sciencecloud:

base_target_dir=/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk/derivatives
base_source_dir=/mnt_03/ds-dnumrisk/derivatives

for sub in $(seq -w 2 66); do
mkdir ${base_target_dir}/dwi_connectome/sub-${sub}
mkdir ${base_target_dir}/dwi_connectome/sub-${sub}/dti
mkdir ${base_target_dir}/dwi_connectome/sub-${sub}/t2

cp ${base_source_dir}/freesurfer/sub-${sub}/mri/aparc+aseg.mgz ${base_target_dir}/dwi_connectome/sub-${sub}/t2/aparc+aseg.mgz 
cp ${base_source_dir}/fmriprep/sub-${sub}/ses-1/anat/sub-${sub}_ses-1_desc-preproc_T1w.nii.gz ${base_target_dir}/dwi_connectome/sub-${sub}/t2/orig_T1.nii.gz
bet ${base_target_dir}/dwi_connectome/sub-${sub}/t2/orig_T1.nii.gz ${base_target_dir}/dwi_connectome/sub-${sub}/t2/T2_SVRTK.nii.gz -f 0.7

cp ${base_target_dir}/dwi_preproc/sub-${sub}/sub-${sub}_dir-all_dwi.nii.gz ${base_target_dir}/dwi_connectome/sub-${sub}/dti/dti.nii.gz
cp ${base_target_dir}/dwi_preproc/sub-${sub}/sub-${sub}_dir-all_dwi.bvals ${base_target_dir}/dwi_connectome/sub-${sub}/dti/bvals
cp ${base_target_dir}/dwi_preproc/sub-${sub}/sub-${sub}_dir-all_dwi.bvecs ${base_target_dir}/dwi_connectome/sub-${sub}/dti/bvecs
; done

- ... <base_dir>/<subj_id>/dti:
                 ./dti.nii.gz (diffusion image)
                  ./bvals # txt file 
                  ./bvecs # txt file

