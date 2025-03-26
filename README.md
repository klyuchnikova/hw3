# Task Final (I made it as a branch from task 3 i know)

### Nagging

So first I would like to note that I have already finished first two punkts in my previous task. However this time I would like to properly add conda env!

I tried to use a python file to build it. I put it in /envs folder. That file may have some inaccuracies, but basically I ran it, it created the config, then simply create env from config, then 

```bash
conda env export -n triton_cv_env310 --file D:\E\Copy\PyCharm\Hometask\ml_hard_models_2025\hw3\envs\triton_cv_env310.env
```

Anyway hopefully you don't require me to actually run docker and connect jupyter to there, it's just... uncomfortable.

Then we start the docker

```bash
docker-compose -f docker-compose.yml up
```

I tried some freaky dances with Dcoker file and making an image to update some gcc libs, but I STILL could not connect to env. I don't understad( I was forces to use pip install but at least I checked that it's all running

I didn't really understand why that happened. I have mild suspicoun that on windows I got a wrong version of conda.

### Results
On test dataset (or rather a sample) we got these results. Seem pretty good - accuracy almost 0.9!
That means everything works and I can go to bed

FP16_Model:
Classification Accuracy: 0.8924

Prediction Precision: 0.9814271749755621

Detection Recall: 0.8006379585326954

Model F1: 0.8818620992534036