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
            flow = np.fromfile(f,np.short).reshape(1024,2048,3)

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


############################################################################
workspace_dir = "./workspace-cityscapes/"
# cityscapes_sequence_root = "/home/hyb/data/cityscapes/leftImg8bit_sequence"
cityscapes_root = "/home/hyb/data/cityscapes/"
cityscapes_sequence_root = os.path.join(cityscapes_root, "leftImg8bit_sequence")
cityscapes_GT_root = os.path.join(cityscapes_root, "gtFine")

ref_gap = 12

encode_fps = 17

bitrate=5000

for key_dist in range(0, ref_gap):

    decoded_output_dir = os.path.join(cityscapes_sequence_root, "%dM-GOP%d/decoded_GOP%d_dist_%d/leftImg8bit/"%(int(bitrate/1000),ref_gap, ref_gap, key_dist))
    
    MV_output_dir = os.path.join(cityscapes_sequence_root, "%dM-GOP%d/MVmap_GOP%d_dist_%d/"%(int(bitrate/1000),ref_gap, ref_gap, key_dist))
    
    frame_output_dir = os.path.join(cityscapes_sequence_root, "%dM-GOP%d/frames/"%(int(bitrate/1000),ref_gap))

    for split in ['train','val']:
        if split != 'val' and key_dist != ref_gap-1:
            continue

        src_dir = cityscapes_GT_root
        dst_dir = os.path.join(
            cityscapes_sequence_root,"%dM-GOP%d/decoded_GOP%d_dist_%d/"%(int(bitrate/1000),ref_gap, ref_gap, key_dist),
            "gtFine")

        if not os.path.exists(decoded_output_dir):
            os.makedirs(decoded_output_dir)

        if not os.path.exists(dst_dir):
            cmd = "ln -s %s %s"%(src_dir, dst_dir)
            # print(cmd)
            # exit(0)
            os.system(cmd)

        for scene in os.listdir(os.path.join(cityscapes_sequence_root, split))[0:]:
            total_list = os.listdir(os.path.join(cityscapes_GT_root, split, scene))
            total_list.sort()

            ###########################################################################

            for fn in total_list:
                ## example: berlin_000271_000019_gtFine_color.png
                if not fn.endswith("_color.png"):
                    continue
                
                prefix = "_".join(fn.split('_')[:2])

                dst_name = "_".join(fn.split('_')[:-1] + ["leftImg8bit.png"])
                
                this_cityscapes_sequence_root = os.path.join(cityscapes_sequence_root, split, scene)
                this_decoded_output_dir = os.path.join(decoded_output_dir, split, scene)
                this_MV_output_dir = os.path.join(MV_output_dir, split, scene)
                this_frame_output_dir = os.path.join(frame_output_dir, split, scene)

                for this_output_dir in [this_decoded_output_dir, this_MV_output_dir, this_frame_output_dir]:
                    if not os.path.exists(this_output_dir):
                        os.makedirs(this_output_dir)

                start_idx = int(fn.split('_')[2])-key_dist
                 
                if not os.path.exists(workspace_dir):
                    os.makedirs(workspace_dir)

                    x265_path = os.path.join(os.getcwd(),"./x265/build/x265")
                    cmd = "ln -s %s %s/x265"%(x265_path, workspace_dir)
                    os.system(cmd)
                    
                    libde265_path = os.path.join(os.getcwd(),"./libde265/build/dec265/dec265")
                    cmd = "ln -s %s %s/dec265"%(libde265_path, workspace_dir)
                    os.system(cmd)

                    # print(name)
                    
                
                # import pdb; pdb.set_trace()
                for iidx in range(start_idx, int(fn.split('_')[2])+1):
                    ## example: berlin_000271_000029_leftImg8bit.png
                    name_split = fn.split('_')[:2] + ["%06d"%iidx, "leftImg8bit.png"]
                    name = "_".join(name_split) 
                    # print(name)

                    orig_path = os.path.join(this_cityscapes_sequence_root,name)
                    dst_path = os.path.join(workspace_dir,"%03d.png"%(iidx-start_idx))
                    cmd = "ln -s %s %s"%(orig_path, dst_path)
                    print(cmd)

                    os.system(cmd)

                # exit(0)
                #############################################################################

                encode_cmd = "ffmpeg -y -i %s"%workspace_dir+"/%03d.png -f rawvideo -pix_fmt yuv420p "+os.path.join(workspace_dir,"proxy.yuv")
                os.system(encode_cmd)

                #############################################################################

                # for bitrate in [2000]:
                
                GOP=(ref_gap)

                cmd = "%s/x265 --input-res 2048x1024 --fps %d --rect --amp --input %s/proxy.yuv --bitrate %d --keyint %d --bframes 0 %s/proxy_%d.hevc"%(workspace_dir,encode_fps,workspace_dir,bitrate, GOP, workspace_dir, bitrate)

                os.system(cmd)

                # exit(0)

                ##############################################################################

                cmd = "%s/dec265 -q %s/proxy_%d.hevc -p %s/"%(workspace_dir,workspace_dir, bitrate, workspace_dir)
                os.system(cmd)

                cmd = "ffmpeg -y -i %s/proxy_%d.hevc %s/"%(workspace_dir, bitrate, workspace_dir) + "decoded-%03d.png"
                os.system(cmd)

                ###############################################################################

                
                ## Decoded annotated frames
                ## ffmpeg decode start from 001
                src_path = os.path.join(workspace_dir,"decoded-%03d.png"%(key_dist+1))

                dst_path = os.path.join(this_decoded_output_dir, dst_name)
                cmd = "cp %s %s"%(src_path, dst_path)
                # import pdb; pdb.set_trace()
                os.system(cmd)

                ## Decoded reference frames
                src_path = os.path.join(workspace_dir,"decoded-001.png")

                dst_name = "_".join(fn.split('_')[:2] + ["%06d"%start_idx, "leftImg8bit.png"]) 
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
                    # exit(0)
                    src_path = os.path.join(workspace_dir,"merged_test_%03d.bin"%key_dist)

                    dst_name =  "_".join(fn.split('_')[:-1] + ["leftImg8bit.bin"])
                    dst_path = os.path.join(this_MV_output_dir, dst_name)
                    cmd = "cp %s %s"%(src_path, dst_path)
                    
                    os.system(cmd)
                    # exit(0)

                #############################################################################
        
                cmd = "rm -r %s"%workspace_dir
                os.system(cmd)



