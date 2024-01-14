from __init__ import *

def config_to_trans(trans_config):
    transform = carla.Transform(carla.Location(trans_config["location"][0],
                                               trans_config["location"][1],
                                               trans_config["location"][2]),
                                carla.Rotation(trans_config["rotation"][0],
                                               trans_config["rotation"][1],
                                               trans_config["rotation"][2]))
    return transform

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k