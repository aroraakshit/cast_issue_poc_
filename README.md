# cast_issue_poc_
### Proof of issue faced in TensorFlow issue [#21192](https://github.com/tensorflow/tensorflow/issues/21192)

TensorFlow: 1.8.0 \
Python: 3.6.4 \
OS: Linux Ubuntu 16.04.4

To export SavedModel:
```python train_model.py --export=True```

To examine the SavedModel: ```saved_model_cli show --dir serve/1533078076/ --all```

To run the tensorflow server: ```tensorflow_model_server --port=9000 --model_name=1533078076 --model_base_path=serve/```

Sample request: ```python3 /home/mldev/intelligent_sp/train_scripts/client.py --input="{'word':'NumberLine', 'drawing':[[[262,262,262,262,262,269,285,315,359,419,493,574,659,790,851,915,965,992,998,971,910,836,778,756,757,777,826,935,1070,1210,1313,1365,1384,1375,1338,1258,1146,1067,1031,1009,999,996,1004,1018,1023,1015,1015],[841,836,824,806,780,750,711,663,604,535,459,386,346,347,380,427,491,552,623,701,799,903,992,1044,1074,1088,1095,1092,1076,1048,1007,961,915,877,841,805,772,766,778,805,833,854,865,870,871,870,870],[7,13,37,54,70,88,104,121,136,153,173,188,205,231,255,272,287,304,321,339,355,372,387,405,421,438,455,471,488,504,521,538,554,572,588,605,622,637,654,672,688,704,722,738,755,797,800]],[[1075,1181,1273,1329,1347,1333,1279,1205,1099,952,740,556,443,414,419,452,510,600,720,806,840,845,847,856,896,975,1065,1121,1147,1156,1151,1103,1017,1017],[704,737,786,837,880,927,981,1021,1049,1070,1070,1078,1106,1136,1152,1164,1172,1172,1172,1169,1167,1168,1180,1217,1271,1319,1319,1319,1317,1303,1265,1171,1056,1056],[0,14,34,47,65,80,98,114,130,148,164,181,198,214,231,247,265,282,298,315,332,348,364,381,399,430,448,464,481,498,514,531,544,547]],[[710,784,864,942,991,1007,1008,996,954,887,790,655,545,501,505,533,596,707,854,957,984,987,987,987],[619,618,627,651,677,695,709,724,747,773,792,814,855,906,942,965,988,1013,1036,1042,1045,1046,1058,1058],[0,16,33,49,67,84,100,117,133,150,167,185,199,217,233,250,284,301,316,333,350,367,390,392]]]}"```

Model files with checkpoints and training.tfrecord.classes can be found [here](https://drive.google.com/drive/folders/1lNjbNwzC7YaNmuuq6PqvLFiUmLXwH2GH?usp=sharing).

