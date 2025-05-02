import os.path as op
from nipype.interfaces.freesurfer import SurfaceTransform

def transform_surfaces(in_file, fs_hemi, bids_folder, source_space, target_space):
    # if native space, put in the subject name as 'sub-XX'
    subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')
    
    def determine_space(subject):
        return subject if subject.startswith('fsaverage') else 'fsnative'
    
    source_space_name = determine_space(source_space)
    target_space_name = determine_space(target_space)
    
    sxfm = SurfaceTransform(subjects_dir=subjects_dir)
    sxfm.inputs.source_file = in_file
    sxfm.inputs.out_file = in_file.replace(source_space_name, target_space_name)
    sxfm.inputs.source_subject = source_space
    sxfm.inputs.target_subject = target_space
    sxfm.inputs.hemi = fs_hemi
    
    r = sxfm.run()
    return r