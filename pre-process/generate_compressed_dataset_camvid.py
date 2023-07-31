import os
import cv2
import numpy as np
from tqdm import tqdm

def mergeMotion(workspace_dir, frame_start, frame_end):
    ref_img = cv2.imread(os.path.join(workspace_dir,'000.png'))
    h, w = ref_img.shape[:2]

    max_ref_num = 3
    
    dp = np.ones([h, w, frame_end + 1, 3], dtype=np.int32) * -1
    
    for f1 in range(frame_start + 1, frame_end + 1):
        print("frame " + str(f1))
        with open(os.path.join(workspace_dir,'test_%03d.bin'%f1)) as f:
            flow = np.fromfile(f,np.short).reshape(720,960,3)

            # print(np.unique(flow[..., 2]))
            intra_mask = np.logical_or(flow[...,2]<0, flow[...,2]>=max_ref_num)
            for i in range(3):
                flow[...,i][intra_mask] = 0
            # print(np.unique(flow[..., 2]))

            k1_mat, j1_mat = np.meshgrid(range(w), range(h))
            j2_mat = j1_mat + np.round(flow[..., 1] / 4).astype(int)
            k2_mat = k1_mat + np.round(flow[..., 0] / 4).astype(int)
            f2_mat = np.maximum(0, f1 - flow[..., 2] - 1)


            # print(np.unique(f2_mat))

            j2_mat = np.clip(j2_mat, 0, h-1)
            k2_mat = np.clip(k2_mat, 0, w-1)


            dp[j1_mat[..., np.newaxis], k1_mat[..., np.newaxis], np.ones([h, w, 1], dtype=np.int32) * f1] = np.where(
                np.tile((flow[..., 2] == 90)[..., np.newaxis, np.newaxis], (1, 1, 1, 3)), # 如果是90就维持原样
                dp[j1_mat[..., np.newaxis], k1_mat[..., np.newaxis], np.ones([h, w, 1], dtype=np.int32) * f1],
                np.where( # 如果不是90
                    np.tile( # 如果 father 还有 father
                        (dp[j2_mat[..., np.newaxis], k2_mat[..., np.newaxis], f2_mat[..., np.newaxis]][..., 2] != -1)
                        [..., np.newaxis],
                        (1, 1, 1, 3)
                    ),
                    dp[j2_mat[..., np.newaxis], k2_mat[..., np.newaxis], f2_mat[..., np.newaxis]], # 链接到 grandfather
                    np.stack([k2_mat, j2_mat, f2_mat], axis=-1)[..., np.newaxis, :] # 否则链接到 father
                )
            )
    k1_mat, j1_mat = np.meshgrid(range(w), range(h))
    
    # import pdb; pdb.set_trace()
    dp[:, :, 1:, 0] = (dp[:, :, 1:, 0] - np.tile(k1_mat[...,np.newaxis],(1,1,frame_end)))*4
    dp[:, :, 1:, 1] = (dp[:, :, 1:, 1] - np.tile(j1_mat[...,np.newaxis],(1,1,frame_end)))*4
    
    return dp[:, :, :, :2]

scene_length_info = {
    '0001TP': {
        'encoded_start_idx': 31,
        'encoded_end_idx': 3721,
        'dataset_start_idx': 6690,
        'dataset_end_idx': 10380,
    },
    '0006R0': {
        'encoded_start_idx': 932,
        'encoded_end_idx': 3932,
        'dataset_start_idx': 930,
        'dataset_end_idx': 3930,
    },
    '0016E5': {
        'encoded_start_idx': 392,
        'encoded_end_idx': 8642,
        'dataset_start_idx': 390,
        'dataset_end_idx': 8640,
    },
    'Seq05VD': {
        'encoded_start_idx': 32,
        'encoded_end_idx': 5102,
        'dataset_start_idx': 30,
        'dataset_end_idx': 5100,
    }
}

############################################################################
workspace_dir = "./workspace-camvid/"

camvid_sequence_root = "/home/hyb/data/camvid-sequence-new"
orig_dir = os.path.join(camvid_sequence_root,"frames/")

camvid_root = "/home/hyb/data/camvid/CamVid/"

ref_gap = 12

encode_fps = 30

bitrate=3000

for key_dist in range(0,ref_gap):

    decoded_output_dir = os.path.join(camvid_sequence_root,"%dM-GOP%d/decoded_GOP%d_dist_%d/"%(int(bitrate/1000),ref_gap, ref_gap, key_dist))
    MV_output_dir = os.path.join(camvid_sequence_root, "%dM-GOP%d/MVmap_GOP%d_dist_%d/"%(int(bitrate/1000),ref_gap, ref_gap, key_dist))
    frame_output_dir = os.path.join(camvid_sequence_root, "%dM-GOP%d/frames/"%(int(bitrate/1000),ref_gap))

    for split in ['train','val','test']:
        
        ## For training and validation, we only need key_dist = ref_gap-1.
        if split != 'test' and key_dist != ref_gap-1:
            continue
            
        src_dir = os.path.join(camvid_root,"%s_labels_with_ignored"%split)
        dst_dir = os.path.join(decoded_output_dir, "%s_labels_with_ignored"%split)

        if not os.path.exists(decoded_output_dir):
            os.makedirs(decoded_output_dir)

        if not os.path.exists(dst_dir):
            cmd = "ln -s %s %s"%(src_dir, dst_dir)
            # print(cmd)
            # exit(0)
            os.system(cmd)

        name_per_seq = {}

        for scene in os.listdir(orig_dir):
            name_per_seq[scene] = []

        total_list = os.listdir(os.path.join(camvid_root, split))
        for name in total_list:
            seq_name = name.split('_')[0]

            name_per_seq[seq_name].append(name)

        ###########################################################################

        for scene in os.listdir(orig_dir):
            print(scene)
            
            this_orig_dir = os.path.join(orig_dir, scene)
            this_decoded_output_dir = os.path.join(decoded_output_dir, split)
            this_MV_output_dir = os.path.join(MV_output_dir, scene)
            this_frame_output_dir = os.path.join(frame_output_dir, scene)

            for this_output_dir in [this_decoded_output_dir, this_MV_output_dir, this_frame_output_dir]:
                if not os.path.exists(this_output_dir):
                    os.makedirs(this_output_dir)

            image_list = os.listdir(this_orig_dir)
            image_list.sort()

            dataset_decoded_idx_gap = scene_length_info[scene]['dataset_start_idx'] - scene_length_info[scene]['encoded_start_idx']

            begin_idx = scene_length_info[scene]['encoded_start_idx'] - 1

            if scene == 'Seq05VD' and key_dist == 0:
                begin_idx = 2 - 1

            for idx, name in enumerate(image_list[begin_idx:]):

                if not os.path.exists(workspace_dir):
                    # import pdb; pdb.set_trace()
                    os.makedirs(workspace_dir)

                    x265_path = os.path.join(os.getcwd(),"./x265/build/x265")
                    cmd = "ln -s %s %s/x265"%(x265_path, workspace_dir)
                    os.system(cmd)
                    
                    libde265_path = os.path.join(os.getcwd(),"./libde265/build/dec265/dec265")
                    cmd = "ln -s %s %s/dec265"%(libde265_path, workspace_dir)
                    os.system(cmd)
                # print(name)
                
                video_name = name
                scene = name.split('_')[0]

                start_frame_idx = int(name.split('_')[-1][1:-4]) - key_dist

                start_frame_split = video_name.split('_')

                dst_name_split = start_frame_split.copy()

                if scene == '0001TP' or scene == '0016E5':
                    dst_idx = int(name.split('_')[-1][:-4]) + dataset_decoded_idx_gap
                elif scene == '0006R0' or scene == 'Seq05VD':
                    dst_idx = int(name.split('_')[-1][1:-4]) + dataset_decoded_idx_gap
                
                if scene == '0001TP':
                    dst_name_split[1] = "%06d.png"%dst_idx
                elif scene == '0006R0' or scene == 'Seq05VD':
                    dst_name_split[1] = "f%05d.png"%dst_idx
                elif scene == '0016E5':
                    dst_name_split[1] = "%05d.png"%dst_idx
                else:
                    # print(scene)
                    exit(1) 
                
                dst_name = "_".join(dst_name_split)

                # print("dst:", dst_name)

                if not dst_name in name_per_seq[scene]:
                    continue
                
                # import pdb; pdb.set_trace()
                for iidx in range(begin_idx+idx-key_dist, begin_idx+idx+(ref_gap-key_dist)):
                    name = image_list[iidx]
                    # print(name)

                    this_frame_idx = int(name.split('_')[-1][1:-4])

                    orig_path = os.path.join(this_orig_dir,name)
                    dst_path = os.path.join(workspace_dir,"%03d.png"%(this_frame_idx-start_frame_idx))
                    cmd = "ln -s %s %s"%(orig_path, dst_path)
                    print(cmd)

                    os.system(cmd)

                #############################################################################
                ### color space conversion mentioned in the supp material
                ### save the individual png into yuv420 format
                ### will the performance get slightly better if yuv 444?
                encode_cmd = "ffmpeg -y -i %s"%workspace_dir+"/%03d.png -f rawvideo -pix_fmt yuv420p "+os.path.join(workspace_dir,"proxy.yuv")
                os.system(encode_cmd)
                #############################################################################

                # for bitrate in [2000]:
                
                GOP=(ref_gap)
                ### save the individual
                ### x265 encoding process, encoded the saved YUV from previous step into a hevc format ( bframes is set to 0 here, keyint is equal to GOP, L)
                cmd = "%s/x265 --input-res 960x720 --fps %d --rect --amp --input %s/proxy.yuv --bitrate %d --keyint %d --bframes 0 %s/proxy_%d.hevc"%(workspace_dir,encode_fps,workspace_dir,bitrate, GOP, workspace_dir, bitrate)
                os.system(cmd)

                # exit(0)

                ##############################################################################

                ### x265 decoding process, convert hevc bitstreams into a mov, mp4, video statistic, B, P, I frames etc (to calculate motion vectors) 
                ### -p is preset @ paper table 4 
                ### -q is the Constant QP rate control @ paper table 4 
                ### q and p here does't matter caused it is defined during encoding process
                cmd = "%s/dec265 -q %s/proxy_%d.hevc -p %s/"%(workspace_dir,workspace_dir, bitrate, workspace_dir)
                os.system(cmd)
                
                ### x265 decoding process, convert hevc bitstreams into the single PNG image files (lossly)
                cmd = "ffmpeg -y -i %s/proxy_%d.hevc %s/"%(workspace_dir, bitrate, workspace_dir) + "decoded-%03d.png"
                os.system(cmd)

                ###############################################################################
                # print(dst_name)
                # exit(0)
                
                ## Decoded annotated frames
                ## ffmpeg decode start from 001
                src_path = os.path.join(workspace_dir,"decoded-%03d.png"%(key_dist+1))

                dst_path = os.path.join(this_decoded_output_dir, dst_name)
                cmd = "cp %s %s"%(src_path, dst_path)
                
                os.system(cmd)

                ## Decoded reference frames
                src_path = os.path.join(workspace_dir,"decoded-001.png")

                dst_name = image_list[begin_idx+idx-key_dist]
                dst_path = os.path.join(this_frame_output_dir, dst_name)

                cmd = "cp %s %s"%(src_path, dst_path)
                
                os.system(cmd)

                # import pdb; pdb.set_trace()
                if key_dist != 0:
                    ## Motion Vectors
                    merged_motion = mergeMotion(workspace_dir, 0, key_dist)
            
                    for end_idx in range(0, key_dist + 1):
                        save_array = merged_motion[...,end_idx,:].astype(np.short)
                        save_array.tofile(os.path.join(workspace_dir,"merged_test_%03d.bin"%end_idx))
                    
                    src_path = os.path.join(workspace_dir,"merged_test_%03d.bin"%key_dist)

                    dst_name = "_".join(dst_name_split)
                    dst_name = dst_name[:-4] + '.bin'
                    dst_path = os.path.join(this_MV_output_dir, dst_name)
                    cmd = "cp %s %s"%(src_path, dst_path)
                    
                    os.system(cmd)
                    # exit(0)

                #############################################################################
        
                cmd = "rm -r %s"%workspace_dir
                os.system(cmd)

    

