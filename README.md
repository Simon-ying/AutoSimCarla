# AutoSimCarla

*A KITTI like Autonomous Driving Dataset simulating by Carla*

<p align='center'>
    <img src="/examples/example.gif" alt="drawing" width="800"/>
</p>

## Preparation

- Install [CARLA](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation) and related packages in requirements.txt.
- change CARLA path in `__init__.py` according to your installation.

![](examples/carla_path.png)



## Scene settings

modify `config.yaml`

```yaml
CARLA_CONFIG:                           # Carla setting
  NUM_OF_VEHICLES: 100                  # Number of vehicles adding to the scene
  AGENT: Behavior                       # Auto-pilot behavior
  BEHAVIOR: normal                      # Auto-pilot behavior

AGENT_CONFIG:                           # Ego-Vechile setting
  SPAWN_ID: 50                          # Starting point in the world
  DEST_ID: 30                           # Destination point in the world
  BLUEPRINT: vehicle.lincoln.mkz_2017   # Vechile model

SENSOR_CONFIG:                          # Sensor settings: transform, sensor name, attribute
  RGB:
    TRANSFORM: {location: [0, 0, 1.8], rotation: [0, 0, 0]}
    BLUEPRINT: sensor.camera.rgb
    ATTRIBUTE: {image_size_x: 1080, image_size_y: 720, fov: 90}

  ...

FILTER_CONFIG:                          # Filtering setting
  ACTOR_FILTER: vehicle.*
  MAX_RENDER_DEPTH_IN_METERS: 50
  MIN_VISIBLE_VERTICES_FOR_RENDER: 3
  MAX_OUT_VERTICES_FOR_RENDER: 5

DISPLAY_CONFIG:                         # Visualization setting
  WIDTH: 1080
  HEIGHT: 720

SAVE_CONFIG:                            # Saving path
  ROOT_PATH: data/00

MAP: Town10HD_Opt                       # World map
```


## Run code

```bash
python automatic_control.py             # Generate KITTI like dataset
python visulization.py                  # Visualization and also an example for data processing
```

## Issues

In CARLA 0.9.15, `carla.CityObjectLable.Vechiles` changes to `carla.CityObjectLabel.Car`, `carla.CityObjectLabel.Truck`, `carla.CityObjectLabel.Bus`.

<!-- ## References
[Lidar data with motion distortion](http://asrl.utias.utoronto.ca/datasets/mdlidar/index.html) -->
