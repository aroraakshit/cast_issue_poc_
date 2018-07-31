# cast_issue_poc_
### Proof of issue faced in TensorFlow issue [#21192](https://github.com/tensorflow/tensorflow/issues/21192)

TensorFlow: 1.8.0 \
Python: 3.6.4

To export SavedModel:
```python train_model.py --export=True```

To examine the SavedModel: ```saved_model_cli show --dir serve/1533078076/ --all```

Model files with checkpoints and training.tfrecord.classes can be found [here](https://drive.google.com/drive/folders/1lNjbNwzC7YaNmuuq6PqvLFiUmLXwH2GH?usp=sharing).

