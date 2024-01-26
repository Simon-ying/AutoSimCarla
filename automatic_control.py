"""Generate dataset"""

from __future__ import print_function
from __init__ import *

from export_utils import *
from world import World
from display import HUD, KeyboardControl
from data_saver import DataSave
from utils import get_actor_blueprints, camera_intrinsic

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def eulerAngles2rotationMat(theta):
    theta = [i * math.pi / 180 for i in theta]
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = R_x.dot(R_y.dot(R_z))
    return R

  
    
# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None
    with open("configs.yaml") as f:
        config = yaml.safe_load(f)
    dsv = DataSave(config)
    OUTPUT_FOLDER = dsv.OUTPUT_FOLDER
    
    # List of moving objects
    vehicles_list = []

    try:
        # nrandom.seed(args.seed)
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.load_world(config["MAP"])

        # ----------------
        # Set synchro mode
        # ----------------
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        sim_world.apply_settings(settings)
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # -----------
        # Set viewers
        # ------------
        spectator = sim_world.get_spectator()

        width = config["DISPLAY_CONFIG"]["WIDTH"]
        height = config["DISPLAY_CONFIG"]["HEIGHT"]
        display = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(width, height)

        intrinsic = camera_intrinsic(width, height)

        world = World(client.get_world(), hud, config)
        controller = KeyboardControl(world)
        if config["CARLA_CONFIG"]["AGENT"] == "Basic":
            agent = BasicAgent(world.player)
        else:
            agent = BehaviorAgent(world.player, 
                                  behavior=config["CARLA_CONFIG"]["BEHAVIOR"])

        # ----------------------
        # Set agenet destination
        # ----------------------
        SPAWN_ID = config["AGENT_CONFIG"]["SPAWN_ID"]
        DEST_ID = config["AGENT_CONFIG"]["DEST_ID"]
        spawn_points = world.map.get_spawn_points()
        destination = spawn_points[DEST_ID].location
        agent.set_destination(destination)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # TODO: only add 4 wheels vehicles now
        blueprints = get_actor_blueprints(sim_world, "vehicle.*", "All")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        spawn_points = world.map.get_spawn_points()[:SPAWN_ID] + world.map.get_spawn_points()[SPAWN_ID+1:]
        random.shuffle(spawn_points)
        for n, transform in enumerate(spawn_points):
            if n >= config["CARLA_CONFIG"]["NUM_OF_VEHICLES"]:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        all_vehicle_actors = world.world.get_actors(vehicles_list)
        vehicle_lables = [carla.CityObjectLabel.Car, carla.CityObjectLabel.Truck, carla.CityObjectLabel.Bus]
        environment_actors = []
        for lable in vehicle_lables:
            environment_actors.extend(world.world.get_environment_objects(carla.CityObjectLabel.Car)) # carla.CityObjectLabel.Vehicles 0.9.13
        # Set automatic vehicle lights update if specified
        """
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)
        """
        # TODO: add pedestrians

        clock = pygame.time.Clock()
        print('spawned %d vehicles, press Ctrl+C to exit.' % (len(vehicles_list)))

        frames = 0
        init = False
        while True:
            clock.tick()
            world.world.tick()

            if controller.parse_events():
                return
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # set spectator above the vehicle
            spec_transform = world.player.get_transform()
            spectator.set_transform(carla.Transform(spec_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
        
            if agent.done():
                print("The target has been reached, stopping the simulation")
                break
            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)
            
            try:
                # Get the data once it's received.
                lidar_data = world.lidar.queue.get(True, 1.0)
                camera_data = world.rgb_camera.queue.get(True, 1.0)
                depth_data = world.depth_camera.queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            try:
                # Calculate lidar pose
                lidar_rotation = world.lidar.sensor.get_transform().rotation
                lidar_location = world.lidar.sensor.get_transform().location
                rotationMat = eulerAngles2rotationMat([lidar_rotation.roll, -lidar_rotation.pitch, lidar_rotation.yaw])
                translation = np.array([lidar_location.x, -lidar_location.y, lidar_location.z]).reshape((3,1))
                if not init:
                    H_init = np.c_[rotationMat, translation]
                    H_init = np.r_[H_init, np.array([[0, 0, 0, 1]])]
                    pre_information = {}
                    init = True
                H = np.c_[rotationMat, translation]
                H = np.r_[H, np.array([[0, 0, 0, 1]])]
                H = np.linalg.inv(H_init).dot(H)

                extrinsic = np.mat(world.rgb_camera.sensor.get_transform().get_matrix())
                world_2_lidar = np.array(world.lidar.sensor.get_transform().get_inverse_matrix())

                vehicle_actors = []
                for obj in all_vehicle_actors:
                    temp_actor = {}
                    temp_actor["transform"] = obj.get_transform()
                    temp_actor["bounding_box"] = obj.bounding_box
                    temp_actor["type"] = obj_type(obj)
                    temp_actor["velocity"] = obj.get_velocity()
                    temp_actor["acceleration"] = obj.get_acceleration()
                    temp_actor["angular_velocity"] = obj.get_angular_velocity()
                    temp_actor["id"] = obj.id
                    vehicle_actors.append(temp_actor)
                for obj in environment_actors:
                    temp_actor = {}
                    temp_actor["transform"] = obj.transform
                    temp_actor["bounding_box"] = obj.bounding_box
                    temp_actor["type"] = "Car"
                    temp_actor["velocity"] = carla.Vector3D(0., 0., 0.)
                    temp_actor["acceleration"] = carla.Vector3D(0., 0., 0.)
                    temp_actor["angular_velocity"] = carla.Vector3D(0., 0., 0.)
                    temp_actor["id"] = obj.id
                    vehicle_actors.append(temp_actor)


                agent_player = {}
                agent_player["transform"] = world.player.get_transform()
                # save ground truth
                

                if frames >= 10:
                    if pre_information:               
                        dsv.save_training_files(lidar_data, camera_data, depth_data, pre_information)
                        with open(OUTPUT_FOLDER + "/poses.txt",'a') as f:
                            np.savetxt(f, np.reshape(H[:3, :].ravel(), (1,12)))
                        with open(OUTPUT_FOLDER + "/times.txt", 'a') as f:
                            f.writelines(str(lidar_data.timestamp)+"\n")
                    
                    pre_information["actors"] = vehicle_actors
                    pre_information["agent"] = agent_player
                    pre_information["world_2_lidar"] = world_2_lidar
                    pre_information["extrinsic"] = extrinsic
                    pre_information["intrinsic"] = intrinsic
                

            except Exception as e:
                print(e)
                break
            
            frames += 1


    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()
            print('\ndestroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        pygame.quit()



# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Debug mode'
    )
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_level = logging.WARNING
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    
    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
