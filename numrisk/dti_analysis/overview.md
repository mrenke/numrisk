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

