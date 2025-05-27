from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from create_annotation import create_annotations
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--st_rd", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1203)
    parser.add_argument("--div", type=int, default=3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # Load BiRefNet with weights
    from transformers import AutoModelForImageSegmentation
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)

    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to(device)
    birefnet.eval()

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


    lvis_class_list = ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium', 'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry', 'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'compass', 'computer_keyboard', 'condiment', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie', 'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headset', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah', 'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate', 'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument', 'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parchment', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet', 'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat', 'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map', 'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_cap', 'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)', 'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini']


    oiv7_class_list = ['accordion', 'adhesive_tape', 'aircraft', 'alarm_clock', 'alpaca', 'ambulance', 'animal', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke', 'auto_part', 'axe', 'backpack', 'bagel', 'baked_goods', 'balance_beam', 'ball_(object)', 'balloon', 'banana', 'band-aid', 'banjo', 'barge', 'barrel', 'baseball_bat', 'baseball_glove', 'bat_(animal)', 'bathroom_accessory', 'bathroom_cabinet', 'bathtub', 'beaker', 'bear', 'beard', 'bed', 'bee', 'beehive', 'beer', 'beetle', 'bell_pepper', 'belt', 'bench', 'bicycle', 'bicycle_helmet', 'bicycle_wheel', 'bidet', 'billboard', 'billiard_table', 'binoculars', 'bird', 'blender', 'blue_jay', 'boat', 'bomb', 'book', 'bookcase', 'boot', 'bottle', 'bottle_opener', 'bow_and_arrow', 'bowl', 'bowling_equipment', 'box', 'boy', 'brassiere', 'bread', 'briefcase', 'broccoli', 'bronze_sculpture', 'brown_bear', 'building', 'bull', 'burrito', 'bus', 'bust', 'butterfly', 'cabbage', 'cabinetry', 'cake', 'cake_stand', 'calculator', 'camel', 'camera', 'can_opener', 'canary', 'candle', 'candy', 'cannon', 'canoe', 'cantaloupe', 'car', 'carnivore', 'carrot', 'cart', 'cassette_deck', 'castle', 'cat', 'cat_furniture', 'caterpillar', 'cattle', 'ceiling_fan', 'cello', 'centipede', 'chainsaw', 'chair', 'cheese', 'cheetah', 'chest_of_drawers', 'chicken', 'chime', 'chisel', 'chopsticks', 'christmas_tree', 'clock', 'closet', 'clothing', 'coat', 'cocktail', 'cocktail_shaker', 'coconut', 'coffee_(drink)', 'coffee_cup', 'coffee_table', 'coffeemaker', 'coin', 'common_fig', 'common_sunflower', 'computer_keyboard', 'computer_monitor', 'computer_mouse', 'container', 'convenience_store', 'cookie', 'cooking_spray', 'corded_phone', 'cosmetics', 'couch', 'countertop', 'cowboy_hat', 'crab', 'cream', 'cricket_ball', 'crocodile', 'croissant', 'crown', 'crutch', 'cucumber', 'cupboard', 'curtain', 'cutting_board', 'dagger', 'dairy_product', 'deer', 'desk', 'dessert', 'diaper', 'dice', 'digital_clock', 'dinosaur', 'dishwasher', 'dog', 'dog_bed', 'doll', 'dolphin', 'door', 'door_handle', 'doughnut', 'dragonfly', 'drawer', 'dress', 'drill_(tool)', 'drink', 'drinking_straw', 'drum', 'duck', 'dumbbell', 'eagle', 'earring', 'egg', 'elephant', 'envelope', 'eraser', 'face_powder', 'facial_tissue_holder', 'falcon', 'fashion_accessory', 'fast_food', 'fax', 'fedora', 'filing_cabinet', 'fire_hydrant', 'fireplace', 'fish', 'fixed-wing_aircraft', 'flag', 'flashlight', 'flower', 'flowerpot', 'flute', 'flying_disc', 'food', 'food_processor', 'football', 'football_helmet', 'footwear', 'fork', 'fountain', 'fox', 'french_fries', 'french_horn', 'frog', 'fruit', 'frying_pan', 'furniture', 'garden_asparagus', 'gas_stove', 'giraffe', 'girl', 'glasses', 'glove', 'goat', 'goggles', 'goldfish', 'golf_ball', 'golf_cart', 'gondola', 'goose', 'grape', 'grapefruit', 'grinder', 'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger', 'hammer', 'hamster', 'hand_dryer', 'handbag', 'handgun', 'harbor_seal', 'harmonica', 'harp', 'harpsichord', 'hat', 'headphones', 'heater', 'hedgehog', 'helicopter', 'helmet', 'high_heels', 'hiking_equipment', 'hippopotamus', 'home_appliance', 'honeycomb', 'horizontal_bar', 'horse', 'hot_dog', 'house', 'houseplant', 'human_arm', 'human_body', 'human_ear', 'human_eye', 'human_face', 'human_foot', 'human_hair', 'human_hand', 'human_head', 'human_leg', 'human_mouth', 'human_nose', 'humidifier', 'ice_cream', 'indoor_rower', 'infant_bed', 'insect', 'invertebrate', 'ipod', 'isopod', 'jacket', 'jacuzzi', 'jaguar_(animal)', 'jeans', 'jellyfish', 'jet_ski', 'jug', 'juice', 'kangaroo', 'kettle', 'kitchen_&_dining_room_table', 'kitchen_appliance', 'kitchen_knife', 'kitchen_utensil', 'kitchenware', 'kite', 'knife', 'koala', 'ladder', 'ladle', 'ladybug', 'lamp', 'land_vehicle', 'lantern', 'laptop', 'lavender_(plant)', 'lemon_(plant)', 'leopard', 'light_bulb', 'light_switch', 'lighthouse', 'lily', 'limousine', 'lion', 'lipstick', 'lizard', 'lobster', 'loveseat', 'luggage_and_bags', 'lynx', 'magpie', 'mammal', 'man', 'mango', 'maple', 'maraca', 'marine_invertebrates', 'marine_mammal', 'measuring_cup', 'mechanical_fan', 'medical_equipment', 'microphone', 'microwave_oven', 'milk', 'miniskirt', 'mirror', 'missile', 'mixer', 'mixing_bowl', 'mobile_phone', 'monkey', 'moths_and_butterflies', 'motorcycle', 'mouse', 'muffin', 'mug', 'mule', 'mushroom', 'musical_instrument', 'musical_keyboard', 'nail_(construction)', 'necklace', 'nightstand', 'oboe', 'office_building', 'office_supplies', 'orange_(fruit)', 'organ_(musical_instrument)', 'ostrich', 'otter', 'oven', 'owl', 'oyster', 'paddle', 'palm_tree', 'pancake', 'panda', 'paper_cutter', 'paper_towel', 'parachute', 'parking_meter', 'parrot', 'pasta', 'pastry', 'peach', 'pear', 'pen', 'pencil_case', 'pencil_sharpener', 'penguin', 'perfume', 'person', 'personal_care', 'personal_flotation_device', 'piano', 'picnic_basket', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pitcher_(container)', 'pizza', 'pizza_cutter', 'plant', 'plastic_bag', 'plate', 'platter', 'plumbing_fixture', 'polar_bear', 'pomegranate', 'popcorn', 'porch', 'porcupine', 'poster', 'potato', 'power_plugs_and_sockets', 'pressure_cooker', 'pretzel', 'printer', 'pumpkin', 'punching_bag', 'rabbit', 'raccoon', 'racket', 'radish', 'ratchet_(device)', 'raven', 'rays_and_skates', 'red_panda', 'refrigerator', 'remote_control', 'reptile', 'rhinoceros', 'rifle', 'ring_binder', 'rocket', 'roller_skates', 'rose', 'rugby_ball', 'ruler', 'salad', 'salt_and_pepper_shakers', 'sandal', 'sandwich', 'saucer', 'saxophone', 'scale', 'scarf', 'scissors', 'scoreboard', 'scorpion', 'screwdriver', 'sculpture', 'sea_lion', 'sea_turtle', 'seafood', 'seahorse', 'seat_belt', 'segway', 'serving_tray', 'sewing_machine', 'shark', 'sheep', 'shelf', 'shellfish', 'shirt', 'shorts', 'shotgun', 'shower', 'shrimp', 'sink', 'skateboard', 'ski', 'skirt', 'skull', 'skunk', 'skyscraper', 'slow_cooker', 'snack', 'snail', 'snake', 'snowboard', 'snowman', 'snowmobile', 'snowplow', 'soap_dispenser', 'sock', 'sofa_bed', 'sombrero', 'sparrow', 'spatula', 'spice_rack', 'spider', 'spoon', 'sports_equipment', 'sports_uniform', 'squash_(plant)', 'squid', 'squirrel', 'stairs', 'stapler', 'starfish', 'stationary_bicycle', 'stethoscope', 'stool', 'stop_sign', 'strawberry', 'street_light', 'stretcher', 'studio_couch', 'submarine', 'submarine_sandwich', 'suit', 'suitcase', 'sun_hat', 'sunglasses', 'surfboard', 'sushi', 'swan', 'swim_cap', 'swimming_pool', 'swimwear', 'sword', 'syringe', 'table', 'table_tennis_racket', 'tablet_computer', 'tableware', 'taco', 'tank', 'tap', 'tart', 'taxi', 'tea', 'teapot', 'teddy_bear', 'telephone', 'television', 'tennis_ball', 'tennis_racket', 'tent', 'tiara', 'tick', 'tie', 'tiger', 'tin_can', 'tire', 'toaster', 'toilet', 'toilet_paper', 'tomato', 'tool', 'toothbrush', 'torch', 'tortoise', 'towel', 'tower', 'toy', 'traffic_light', 'traffic_sign', 'train', 'training_bench', 'treadmill', 'tree', 'tree_house', 'tripod', 'trombone', 'trousers', 'truck', 'trumpet', 'turkey', 'turtle', 'umbrella', 'unicycle', 'van', 'vase', 'vegetable', 'vehicle', 'vehicle_registration_plate', 'violin', 'volleyball_(ball)', 'waffle', 'waffle_iron', 'wall_clock', 'wardrobe', 'washing_machine', 'waste_container', 'watch', 'watercraft', 'watermelon', 'weapon', 'whale', 'wheel', 'wheelchair', 'whisk', 'whiteboard', 'willow', 'window', 'window_blind', 'wine', 'wine_glass', 'wine_rack', 'winter_melon', 'wok', 'woman', 'wood-burning_stove', 'woodpecker', 'worm', 'wrench', 'zebra', 'zucchini']

    # Function to add articles "a" or "an" before each class name
    def add_article(cls_name):
        vowels = 'AEIOUaeiou'
        if cls_name[0] in vowels:
            return f"an {cls_name}"
        else:
            return f"a {cls_name}"


    # Replace underscores with spaces and add articles
    # formatted_classes = [add_article(cls.replace("_", " ")) for cls in lvis_class_list]
    # formatted_classes = [add_article(cls.replace("_", " ")) for cls in oiv7_class_list]

    # JSON template for each class
    template = [
        "a photo of {}",
        "a realistic photo of {}",
        "a photo of {} in pure background",
        "{} in a white background",
        "{} without background",
        "{} isolated on white background",
        "a photo of {} in a white background"
    ]

    def extract_object(birefnet, imagepath):
        # Data settings
        # image_size = (1024, 1024)
        image_size = (512, 512)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        while not os.path.exists(imagepath):
            time.sleep(1)

        image = Image.open(imagepath)
        input_images = transform_image(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image, mask

    
    def filter_by_clip(classPrompt, image, clip, clipProcessor, device):

        whiteBackground = Image.new("RGBA", image.size, (255, 255, 255, 255))
        blackBackground = Image.new("RGBA", image.size, (0, 0, 0, 255))

        image_w = Image.alpha_composite(whiteBackground, image)
        image_b = Image.alpha_composite(blackBackground, image)


        inputs = clipProcessor(text=[classPrompt], images=[image_w, image_b], return_tensors="pt", padding=True).to(device)
        outputs = clip(**inputs)
        logits_per_image = outputs.logits_per_image
        # print(logits_per_image)
        # print(max([logits_per_image[0][0].item(), logits_per_image[1][0].item()]))
        if max([logits_per_image[0][0].item(), logits_per_image[1][0].item()]) < 25:
            return True
        return False

    # pathFormat = "/data2/objdet/fg_lvis/{}/{}"
    # writePathFormat = "/data2/objdet/fg_mask/{}_mask/{}{}"

    # pathFormat = "/data3/objdet/fg_oiv7/{}/{}"
    # writePathFormat = "/data3/objdet//fg_oiv7_mask/{}_mask/{}{}"

    pathFormat = "/data3/objdet/lvis_feedback_sd3/raw"
    writePathFormat = "/data3/objdet/lvis_feedback_sd3/segmented"
    st_rd = args.st_rd
    numRounds = 30
    # for each round segment objects
    for rd in range(st_rd, numRounds):
        numSelected = 0
        rawRoundPath = pathFormat

        if args.div == -1:
            rawMetaPath = rawRoundPath + "/meta{}.txt".format(str(rd))
            print("Wait for {}".format(rawMetaPath))
            while not os.path.exists(rawMetaPath):
                time.sleep(1)
            f = open("/data3/objdet/lvis_feedback_sd3/raw/meta{}.txt".format(str(rd)), 'r')
            numGenList = ast.literal_eval(f.read())
            f.close()

        else:
            numGenList = [0]*1203
            for i in range(args.div):
                rawMetaPath = rawRoundPath + "/{}meta{}.txt".format(str(i), str(rd))
                print("Wait for {}".format(rawMetaPath))
                while not os.path.exists(rawMetaPath):
                    time.sleep(1)
                f = open("/data3/objdet/lvis_feedback_sd3/raw/{}meta{}.txt".format(str(i), str(rd)), 'r')
                tempNumGenList = ast.literal_eval(f.read())
                f.close()
                numGenList = [sum(x) for x in zip(numGenList, tempNumGenList)]

        if not os.path.exists("/data3/objdet/lvis_feedback_sd3/segmented"): 
            os.mkdir("/data3/objdet/lvis_feedback_sd3/segmented")

        # if not os.path.exists("/data3/objdet/lvis_feedback_sd3/segmented/rd{}".format(str(0))): 
        #     os.mkdir("/data3/objdet/lvis_feedback_sd3/segmented/rd{}".format(str(0)))

        numSelList = [[] for x in range(len(numGenList))]

        # get number of generated object per class
        for numGenIdx in range(len(numGenList)):
            
            numGen = numGenList[numGenIdx]
            className = f'{numGenIdx +1:04}'
            classPrompt = template[0].format(lvis_class_list[numGenIdx])
            # segment images
            for genIdx in range(1,numGen+1):
                rdNum = f'{rd:02}'
                classGenNum = f'{genIdx:04}'
                imgpath = rawRoundPath + '/' + rdNum + '_' + className + '_' + classGenNum + ".png"
                writePath = writePathFormat + '/' + rdNum + '_'  + className + '_' + classGenNum + ".png"
                # writePath2 = writePathFormat + "/rd" + str(rd) + "/" + className + '_' + classGenNum + 'real' + ".png"
                image, masks = extract_object(birefnet, imagepath = imgpath)

            # select images with clip
                # image.save(writePath2)
                if not filter_by_clip(classPrompt, image, clip, clipProcessor, device):
                    numSelList[numGenIdx].append(imgpath)
                    masks.save(writePath)
                    numSelected +=1

        print(numSelected)
        
        f = open("/data3/objdet/lvis_feedback_sd3/segmented/meta{}.txt".format(str(rd)), 'w')
        f.write(str(numSelList))
        f.close()


        create_annotations(rd)




    # @torch.no_grad()
    # def get_CLIP_score(caption: str, images: list):
    #     logits_per_images = []
    #     for img in batchify(images, 400):
    #         inputs = processor(text=[caption] + voc_texts, images=img, return_tensors="pt", padding=True).to("cuda")
    #         outputs = model(**inputs)
    #         logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #         logits_per_images.append(logits_per_image)
    #     # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    #     return torch.cat(logits_per_images, dim=0)


    # def scores_for_one_caption(caption: Path):
    #     keep_files = 4 # 30
    #     images = []
    #     for image in caption.iterdir(): # eg 1.png
    #         try:
    #             images.append(Image.open(image))
    #         except:
    #             pass # weird generation error
    #     scores = get_CLIP_score(caption.stem, images) # (#images, 22)

    #     # 1. select top keep_files*2 lowest consistent_with_voc_labels
    #     consistent_with_voc_labels = scores[:, 1:].max(1).values
    #     double_keep_files = min(keep_files * 2, scores.size(0))
    #     _, indices = torch.topk(-consistent_with_voc_labels.squeeze(), min(double_keep_files, scores.size(0)))
    #     # 2. select top keep_files highest consistent_with_caption
    #     consistent_with_caption = scores[indices, 0]
    #     _, indices = torch.topk(consistent_with_caption, keep_files)
    #     selected_images = [
    #         images[i].filename.split("/")[-1]
    #         for i in indices.detach().cpu().numpy().tolist()
    #     ]
    #     return caption.stem, selected_images



    # # Visualization
    # plt.axis("off")
    # plt.imshow(extract_object(birefnet, imagepath='/data2/objdet/fg_lvis/an avocado/a photo of an avocado in pure background/17.png')[0])
    # plt.show()