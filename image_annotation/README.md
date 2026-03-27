Shipeng Shao/2060907/20160612 Slpit GFP Telomere/20160612 MCP-VN+PCP-VC#024.nd2.tiff

log.csv：源文件路径-源文件名称-tiff名(test1) -csv名 - 时间帧 -可能的Z轴 - 通道

工作流：
按照名称顺序找文件，
视频 划到帧
记录帧数
Image- duplicate 找图
存tiff 图 到raw/ ,命名 test1
进行trackmate找点
存estimated_object_diameter和quality_threshold
存csv 到 raw_csv 命名 test1
存标点图png 到 raw_png 命名 test1
记录到log.csv中