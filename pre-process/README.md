# Pre-processing: Encoding the video frames into compressed videos.

## Installation
1. Install ffmpeg.
    ```
    sudo apt install ffmpeg
    ```
2. Build the x265 library for encoding.
    ```
    cd ./x265/build
    cmake ../source
    make -j
    ```
3. Build the libde265 library for decoding .
    ```
    cd ./libde265
    mkdir build
    cd ./build
    cmake ..
    make -j
    ```

## Pre-processing
### CamVid
1. Download the CamVid dataset: [here](https://www.kaggle.com/datasets/carlolepelaars/camvid?resource=download)
2. Download the original video sequences: [here](http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/)
3. After downloading the files above, your directories should look like:
    ```
    camvid_root
    ┣ test
    ┣ test_labels
    ┣ train
    ┣ train_labels
    ┣ val
    ┣ val_labels
    ┗ class_dict.csv

    camvid_sequence_root
    ┣ 01TP_extract.avi
    ┣ 0005VD.MXF
    ┣ 0006R0.MXF
    ┣ 0016E5.zip.001
    ┗ 0016E5.zip.002
    ```
4. Set ignored labels for CamVid dataset
    ```
    cp ./camvid-pre-process.py /path/to/camvid_root/
    cd /path/to/camvid_root/
    python ./camvid-pre-process.py
    ```
5. Decode the original videos into video frames.
    ```
    cp ./camvid_decode.sh /path/to/camvid_sequence_root/
    cd /path/to/camvid_sequence_root/
    bash ./camvid_decode.sh
    ```
6. Replace the `camvid_root` and `camvid_sequence_root` variables in `./generate_compressed_dataset_camvid.py`. Run:
    ```
    python ./generate_compressed_dataset_camvid.py
    ```
7. After pre-processing, the directory of CamVid should be:
    ```
    camvid_sequence_root
    ┣ 3M-GOP12
    ┣ ┣ decoded_GOP12_dist_0
    ┣ ...
    ┣ ┣ frames
    ┣ ┣ MVmap_GOP12_dist_0
    ┣ ...
    ┣ ┗ MVmap_GOP12_dist_11
    ┣ frames
    ┣ ┣ 0001TP
    ┣ ┣ 0006R0
    ┣ ┣ 0016E5
    ┣ ┗ Seq05VD
    ┣ 01TP_extract.avi
    ┣ 0005VD.MXF
    ┣ 0006R0.MXF
    ┣ 0016E5.zip.001
    ┗ 0016E5.zip.002
    ```


### Cityscapes
1. Download the Cityscapes dataset: [GTFine](https://www.cityscapes-dataset.com/file-handling/?packageID=1) and [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
2. Download the original video sequences: [leftImg8bit_sequence](https://www.cityscapes-dataset.com/file-handling/?packageID=14)
3. After downloading the files above, your directories should look like:
    ```
    cityscapes_root
    ┣ GTFine
    ┣ ┣ test
    ┣ ┣ train
    ┣ ┗ val
    ┣ leftImg8bit
    ┣ ┣ test
    ┣ ┣ train
    ┣ ┗ val
    ┗ leftImg8bit_sequence
      ┣ test
      ┣ train
      ┗ val
    ```
4. Replace the `cityscapes_root` in `./generate_compressed_dataset_cityscapes.py`. Run:
    ```
    python ./generate_compressed_dataset_cityscapes.py
    ```
5. After pre-processing, the directory of Cityscapes should be:
    ```
    cityscapes_root
    ┣ ...
    ┗ leftImg8bit_sequence
      ┣ 5M-GOP12
      ┣ ┣ decoded_GOP12_dist_0
      ┣ ...
      ┣ ┣ frames
      ┣ ┣ MVmap_GOP12_dist_0
      ┣ ...
      ┣ ┗ MVmap_GOP12_dist_11
      ┣ test
      ┣ train
      ┗ val
    ```