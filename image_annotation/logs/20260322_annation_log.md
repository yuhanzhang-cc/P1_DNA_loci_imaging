对于frame、z_axis和 channel的标注，如没有的话统一填0

suntag的图有点一言难尽，可看看效果如何

.mvd2结尾的文件为一个序列，属于Volocity Library的配套文件，与同目录的Data/ 文件夹组成。导入Fiji后会形成一系列图片文件。为区分，annoation_record.csv 中source_path 栏统一写.mvd2文件的路径，在source_file_name 栏写其中的每一个图片名。注意该部分的source_file_name 无法直接检索.

.mvd2导入的每一个图片文件有3个channel（猜测分别为1荧光 2暗场 3明场），并有z轴，猜测为confocal的数据。由于荧光点分布在不同的stacks，单个z axis stack能看到的的荧光很少，因此需要对z轴进行叠加。经过查询，可采用 Image- Stack- Z project- Max Intensity 进行叠加，效果最好。使用该方法的图片在annoation_record.csv的source_file_name 栏前加 MAX_ 予以标注。该部分的z_axis标注为0.

1. Prepare dataset:
deepblink create -i raw_tif/ -l raw_csv_pixel/ -n dataset_128 
|
.npz file

2. Generate config file:
deepblink config
|
.yaml file

3. Modify config.yaml:
 - Set dataset_args.name to dataset.
 - Set savedir to output directory.

4. Train:
deepblink train -c config.yaml -v
|
.h5 file

 pixel_size_map = {
>>     'test1': 1.0,   
>>     'test2': 1.0,
>>     'test3': 0.16,
>>     'test4': 1.0,
>>     'test5': 0.143,
>>     'test6': 0.143
>> }