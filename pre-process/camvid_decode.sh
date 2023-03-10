cat 0016E5.zip* > 0016E5.zip
unzip 0016E5.zip

mkdir frames
mkdir frames/0006R0
mkdir frames/0016E5
mkdir frames/Seq05VD
mkdir frames/0001TP
ffmpeg -i 0005VD.MXF frames/Seq05VD/Seq05VD_%06d.png
ffmpeg -i 0006R0.MXF frames/0006R0/0006R0_%06d.png
ffmpeg -i 01TP_extract.avi frames/0001TP/0001TP_%06d.png
ffmpeg -i 0016E5.MXF frames/0016E5/0016E5_%06d.png