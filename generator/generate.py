import torch
import torchvision
import argparse
import json, os
import numpy as np
import pandas as pd
import time
from diffusers import StableDiffusion3Pipeline
# from diffusers import StableDiffusionPipeline 
from generation_config import GenerationConfig
from load_model import load_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    parser.add_argument("--round", type=int, default=10)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--gPerRd", type=int, default=1203)
    parser.add_argument("--caption_json", default="./data", help="if not '', will only generate DallE images from this json, use `idx` and `scene` to select which to generate")
    parser.add_argument("--num_clusters", default=1, type=int, help="when using RuDalle, split all captions into `num_clusters` chunk and let each machine handle one chunk only")
    parser.add_argument("--output_dir", default="./lvis_data/") # data directory
    parser.add_argument("--cfg_path", default="MRCA_R50_feed.yaml")
    parser.add_argument("--ckpt_dir", default="./ckpt_output/") # checkpoint directory
    parser.add_argument("--st_rd", type=int, default=0)
    parser.add_argument("--div", type=int, default=-1)
    parser.add_argument("--balanced", type=int, default=0)
    parser.add_argument("--val_acc", type=int, default=1)
    args = parser.parse_args()
    if 'PT_DATA_DIR' in os.environ:
        args.output_dir = os.path.join(os.environ['PT_DATA_DIR'], args.output_dir)
    return args


lvis_class_list = ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium', 'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry', 'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'compass', 'computer_keyboard', 'condiment', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie', 'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headset', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah', 'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate', 'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument', 'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parchment', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet', 'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat', 'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map', 'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_cap', 'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)', 'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini']

def calc_cos_between_mats(mat1, mat2):

    distances = torch.nn.functional.pairwise_distance(mat1, mat2, p=2)

    return distances

def get_classifier(ckptDir, ckptPath, cfgPath, ckpt_done_path, rd):

    torch_dtype = torch.float32     
    cos_dist = None
    
    if rd == 0:
        return None, None

    elif rd == 1:
        print("Wait for {}".format(ckpt_done_path))
        while not os.path.exists(ckpt_done_path):
            time.sleep(1)

        ckptPath = ckptDir  + "0.pth"
        fg_classifier = load_model(cfgPath, ckptPath)
        clwDir = ckptDir + "clws"

        if not os.path.exists(clwDir): 
            os.mkdir(clwDir)
        
        curWeight = fg_classifier.roi_heads.box_predictor[-1].cls_score.weight
        # print(curWeight.shape)
        clwPath = clwDir + "/clw{}.pt".format(rd)
        if not os.path.exists(clwPath): 
            torch.save(curWeight, clwPath)



        fg_classifier.eval()
        fg_classifier.cuda()
        fg_classifier.to(torch_dtype)



    else:
        # read classifier at new round
        print("Wait for {}".format(ckpt_done_path))
        while not os.path.exists(ckpt_done_path):
            time.sleep(1)

        fg_classifier = load_model(cfgPath, ckptPath)
        
        clwPath = clwDir + "/clw{}.pt".format(rd)
        prevClwPath = clwDir + "/clw{}.pt".format(rd-1)

        curWeight = fg_classifier.roi_heads.box_predictor[-1].cls_score.weight
        if not os.path.exists(clwPath): 
            torch.save(curWeight, clwPath)
        prevWeight = torch.load(prevClwPath , map_location=torch.device("cpu"))
        curWeightCopy  = curWeight.detach().cpu()

        cos_dist = calc_cos_between_mats(curWeightCopy , prevWeight)


        fg_classifier.eval()
        fg_classifier.cuda()
        fg_classifier.to(torch_dtype)


    return fg_classifier, cos_dist


def generate_single_sample(row, pipe, fg_classifier, fg_preprocessing):
    generator = torch.Generator(device='cuda')
    generator.seed()


    out = pipe(
        prompt=row['prompt'],
        guidance_scale=row['cfg'],
        fg_criterion=row['fg_criterion'],
        fg_scale=row['fg_scale'],
        fg_classifier=fg_classifier,
        fg_preprocessing=fg_preprocessing,
        generator=generator,
        num_inference_steps=30,
        num_images_per_prompt=1,
        cls_index=row['cls_index'],
        guidance_freq=5,
        height=512,
        width=512,)

    return out

def set_genCfg(genCfg, template, numGenList):

    genCfg.reset()

    for idx in range(len(lvis_class_list)):
        for j in range(numGenList[idx]):
            tmpidx = np.random.choice(range(len(template)))  
            prompt = template[tmpidx].format(lvis_class_list[idx])
            genCfg.add_config(prompt = prompt, cls_index = idx+1, inst_index = j+1)


    return


def create_dict_from_file(file_path):
    result_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split by ':'
            parts = line.strip().split(':')
            
            # Check if the line has exactly two parts
            if len(parts) == 2:
                try:
                    key = int(parts[0].strip())
                    value = int(parts[1].strip())
                    result_dict[key] = value
                except ValueError:
                    print(f"Skipping invalid line (non-integer values): {line.strip()}")
            else:
                print(f"Skipping malformed line: {line.strip()}")
                
    return result_dict


def scale_ratios_to_integers(ratios, target_sum):
    # Step 1: Scale ratios to target sum

    scaled = [r * target_sum / sum(ratios) for r in ratios]

    
    # Step 2: Floor the scaled values to get initial integers
    int_values = [int(x) for x in scaled]
    
    # Step 3: Calculate the rounding error
    diff = target_sum - sum(int_values)
    
    # Step 4: Distribute the remaining difference
    # Sort the scaled values based on their decimal parts
    decimal_parts = [(i, scaled[i] - int_values[i]) for i in range(len(ratios))]
    decimal_parts.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(diff):
        idx = decimal_parts[i % len(ratios)][0]
        int_values[idx] += 1

    return int_values


def get_numTrain():
    # get num train per class
    numPerClass = create_dict_from_file("../datasets/metadata/lvisClassInst.txt")

    numTrain = []
    for i in range(len(lvis_class_list)):
        numTrain.append(numPerClass[i+1]) 


    return numTrain



if __name__ == "__main__":
    args = parse_args()

    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) 

    # 1. Set stable diffusion model
   
    access_token = "your_access_token_here"  # Replace with your actual access token

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", token=access_token, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    


    fg_preprocessing = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    

    # prompt template
    template = [
        "a photo of {}",
        "a realistic photo of {}",
        "a photo of {} in pure background",
        "{} in a white background",
        "{} without background",
        "{} isolated on white background",
        "a photo of {} in a white background"
    ]

    genPerClass = 4
    numGen = 1203*genPerClass 
    numGenThird = 1203 // 3

    if args.div == -1:
        genMaskList = [1]*numGen
    elif args.div == 0:    
        genMaskList = [1]*numGenThird
        genMaskList += [0]*numGenThird
        genMaskList += [0]*numGenThird
    elif args.div == 1:
        genMaskList = [0]*numGenThird
        genMaskList += [1]*numGenThird
        genMaskList += [0]*numGenThird
    elif args.div == 2:
        genMaskList = [0]*numGenThird
        genMaskList += [0]*numGenThird
        genMaskList += [1]*numGenThird


    numTrain = get_numTrain()
    numRounds = args.round
    ckptDir = args.ckpt_dir # output dir for reading checkpoints
    imgDir = args.output_dir # output dir for creating images
    cfgPath = args.cfg_path
    st_rd = args.st_rd
    period = 10000
    cos_dist = None
    val_acc = None


    for rd in range(st_rd, numRounds):

        ckptNum = max((rd-1)*period - 1, 0)

    

        ckptPath = ckptDir + "/model_" + str(f'{ckptNum:07}') + ".pth"




        ckpt_done_path = ckptDir + '/rd' + str(rd - 2) + 'done_ckpt.txt'
        if rd == 1:
            ckpt_done_path = ckptDir + '/0' + 'done_ckpt.txt'





        fg_classifier, cos_dist = get_classifier(ckptDir, ckptPath, cfgPath, ckpt_done_path, rd)

        # 2. set number of generated images per class
        if args.balanced == 0:
            numGenRatio = [1 / x for x in numTrain]

            if cos_dist is not None:
                numGenRatio = [a * b for a, b in zip(cos_dist[1:], numGenRatio)]
            
            if args.val_acc and rd>1:
                idxt = (rd-2)*2+1
                with open(ckptDir + 'inference_lvis_v1_val/per_class_mAP{}.json'.format(idxt)) as f:
                    val_acc=list(json.load(f).values())
            
            if val_acc is not None:
                numGenRatio = [a * b for a, b in zip(val_acc, numGenRatio)] 

            numGenList= scale_ratios_to_integers(numGenRatio, numGen)
            
            numGenList= [a * b for a, b in zip(numGenList, genMaskList)]

        else:
            numGenList = numGenList= [a* genPerClass  for a in genMaskList]
            

        # 3. set generation config
        genCfg = GenerationConfig()
        set_genCfg(genCfg, template, numGenList)


        # 4. generate according to config
        if not os.path.exists(imgDir): 
            os.mkdir(imgDir)
        
        if not os.path.exists(imgDir+ "raw"): 
            os.mkdir(imgDir+ "raw")

        if not os.path.exists(imgDir + "raw/rd{}".format(str(0))): 
            os.mkdir(imgDir + "raw/rd{}".format(str(0)))

        for i in range (len(genCfg.config_list)):
            row = genCfg.get_config(i)
            image = generate_single_sample(row, pipe, fg_classifier, fg_preprocessing)

            clsidx = row['cls_index']
            instidx = row['inst_index']
            rawSavePath = imgDir + "raw/{}_{}_{}.png".format(str(f'{rd:02}'), str(f'{clsidx:04}'), str(f'{instidx:04}'))  
            image[0][0].save(rawSavePath)

        # 5. write metadata about generated data
        if args.div == -1:
            f = open(imgDir + "raw/meta{}.txt".format(str(rd)), 'w')
            f.write(str(numGenList))
            f.close()
        else:
            f = open(imgDir + "raw/{}meta{}.txt".format(str(args.div), str(rd)), 'w')
            f.write(str(numGenList))
            f.close()

