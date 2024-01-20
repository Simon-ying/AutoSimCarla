# AutoSimCarla

*A KITTI like Autonomous Driving Dataset simulating by Carla*

<p align='center'>
    <img src="/examples/example.gif" alt="drawing" width="800"/>
</p>

## Preparation

- Install [CARLA](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation) and related packages shown in `__init__.py`
- change CARLA path according to your installation.

![](examples/carla_path.png)



## Scene settings

modify `config.yaml`



## Run code

```bash
python automatic_control.py # Generate KITTI like dataset

python visulization.py # Visualization and also an example for data processing
```

