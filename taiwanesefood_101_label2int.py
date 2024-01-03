def tw101_label2int(label_list):
    label_to_int = {
            'bawan' : 0,
            'beef_noodles' : 1,
            'beef_soup' : 2,
            'bitter_melon_with_salted_eggs' : 3,
            'braised_napa_cabbage' : 4,
            'braised_pork_over_rice' : 5,
            'brown_sugar_cake' : 6,
            'bubble_tea' : 7,
            'caozaiguo' : 8,
            'chicken_mushroom_soup' : 9,
            'chinese_pickled_cucumber' : 10,
            'coffin_toast' : 11,
            'cold_noodles' : 12,
            'crab_migao' : 13,
            'deep-fried_chicken_cutlets' : 14,
            'deep_fried_pork_rib_and_radish_soup' : 15,
            'dried_shredded_squid' : 16,
            'egg_pancake_roll' : 17,
            'eight_treasure_shaved_ice' : 18,
            'fish_head_casserole' : 19,
            'fried-spanish_mackerel_thick_soup' : 20,
            'fried_eel_noodles' : 21,
            'fried_instant_noodles' : 22,
            'fried_rice_noodles' : 23,
            'ginger_duck_stew' : 24,
            'grilled_corn' : 25,
            'grilled_taiwanese_sausage' : 26,
            'hakka_stir-fried' : 27, 
            'hot_sour_soup' : 28,
            'hung_rui_chen_sandwich' : 29,
            'intestine_and_oyster_vermicelli' : 30,
            'iron_egg' : 31,
            'jelly_of_gravey_and_chicken_feet_skin' : 32,
            'jerky' : 33,
            'kung-pao_chicken' : 34,
            'luwei' : 35,
            'mango_shaved_ice' : 36,
            'meat_dumpling_in_chili_oil' : 37,
            'milkfish_belly_congee' : 38,
            'mochi' : 39,
            'mung_bean_smoothie_milk' : 40,
            'mutton_fried_noodles' : 41,
            'mutton_hot_pot' : 42, 
            'nabeyaki_egg_noodles' : 43,
            'night_market_steak' : 44,
            'nougat' : 45,
            'oyster_fritter' : 46, 
            'oyster_omelet' : 47,
            'papaya_milk' : 48,
            'peanut_brittle' : 49,
            'pepper_pork_bun' : 50,
            'pig\'s_blood_soup' : 51,
            'pineapple_cake' : 52,
            'pork_intestines_fire_pot' : 53,
            'potsticker' : 54, 
            'preserved_egg_tofu' : 55,
            'rice_dumpling' : 56,
            'rice_noodles_with_squid' : 57,
            'rice_with_soy-stewed_pork' : 58, 
            'roasted_sweet_potato' : 59,
            'sailfish_stick' : 60,
            'salty_fried_chicken_nuggets' : 61,
            'sanxia_golden_croissants' : 62,
            'saute_spring_onion_with_beef' : 63,
            'scallion_pancake' : 64,
            'scrambled_eggs_with_shrimp' : 65,
            'scrambled_eggs_with_tomatoes' : 66,
            'seafood_congee' :67,
            'sesame_oil_chicken_soup' : 68,
            'shrimp_rice' : 69,
            'sishen_soup' : 70,
            'sliced_pork_bun' : 71,
            'spicy_duck_blood' : 72,
            'steam-fried_bun' : 73,
            'steamed_cod_fish_with_crispy_bean' : 74,
            'steamed_taro_cake' : 75,
            'stewed_pig\'s_knuckles' : 76,
            'stinky_tofu' : 77,
            'stir-fried_calamari_broth' : 78,
            'stir-fried_duck_meat_broth' : 79,
            'stir-fried_loofah_with_clam' : 80,
            'stir-fried_pork_intestine_with_ginger' : 81,
            'stir_fried_clams_with_basil' : 82,
            'sugar_coated_sweet_potato' : 83,
            'sun_cake' : 84,
            'sweet_and_sour_pork_ribs' : 85,
            'sweet_potato_ball' : 86,
            'taiwanese_burrito' : 87,
            'taiwanese_pork_ball_soup' : 88,
            'taiwanese_sausage_in_rice_bun' : 89,
            'tanghulu' : 90,
            'tangyuan' : 91,
            'taro_ball' : 92,
            'three-cup_chicken' : 93,
            'tube-shaped_migao' : 94,
            'turkey_rice' : 95,
            'turnip_cake' : 96,
            'twist_roll' : 97,
            'wheel_pie' : 98,
            'xiaolongbao' : 99,
            'yolk_pastry' : 100
        }
    
    label_list = [label_to_int[label] for label in label_list]

    return label_list